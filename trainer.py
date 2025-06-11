from typing import Any, Callable
from pathlib import Path
import time
import torch
import torch.nn.functional as F
import json
import math
import logging
import numpy as np
import evaluate
from datasets import Dataset
from setfit import Trainer, TrainingArguments, SetFitModel  # , SetFitModel
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from torch.nn.modules import Module
from setfit.trainer import BCSentenceTransformersTrainer
from transformers.trainer_callback import TrainerCallback, IntervalStrategy
from train_args import MiningArguments

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from sentence_transformers.losses import (
    BatchHardTripletLoss,
    BatchHardSoftMarginTripletLoss,
    BatchAllTripletLoss,
    BatchSemiHardTripletLoss,
    BatchHardTripletLossDistanceFunction,
    TripletLoss,
    TripletDistanceMetric,
)
from sentence_transformers.evaluation import BinaryClassificationEvaluator

from util import LOSS_MAP, plot_tsne, print_distance_stats, get_distance_stats, _convert_to_tensor, evaluate_pair_similarities
from sampler import get_samples, random_sampling

logger = logging.getLogger(__name__)

class MiningTrainer(Trainer):

    def __init__(
        self,
        model: SetFitModel | None = None,
        args: MiningArguments | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        model_init: Callable[[], SetFitModel] | None = None,
        metric: str | Callable[[Dataset, Dataset], dict[str, float]] = "accuracy",
        metric_kwargs: dict[str, Any] = None,
        callbacks: list[TrainerCallback] = None,
        column_mapping: dict[str, str] | None = None,
        save_embeddings: bool = False,
    ):

        super().__init__(
            model,
            args,
            train_dataset,
            eval_dataset,
            model_init,
            metric,
            metric_kwargs,
            callbacks,
            column_mapping,
        )

        # print(self.args)

        self.rng = np.random.default_rng(self.args.seed)
        # quick fix to guarantee same random positives for different negative sampling methods/num sampling epochs
        # should be same positive randoms now, except of course if positives get deleted due to no semi/hard negative being available
        self.n_rng = np.random.default_rng(self.args.seed)# if args.positive_sampling_strategy == "random" else None

        self.distances = []
        self.positive_pairs = []
        self.negative_pairs = []
        self.random_pos_pairs = []
        self.random_neg_pairs = []
        
        self.save_embeddings = save_embeddings

    def get_dataset(
        self,
        x: list[str],
        y: list[int] | list[list[int]],
        args: MiningArguments,
        max_pairs: 1 = -1,
    ) -> tuple[Dataset, Module, int, int]:

        if args.sampling_strategy != "mining":
            return super().get_dataset(x, y, args, max_pairs)

        if args.loss in [
            BatchAllTripletLoss,
            BatchHardTripletLoss,
            BatchSemiHardTripletLoss,
            BatchHardSoftMarginTripletLoss,
        ]:
            logger.warning(
                f"{LOSS_MAP[args.loss]} not compatible with sample mining. Defaulting to num_iter=20!"
            )
            setfit_args = args.get_setfit_args()
            setfit_args.num_iterations = 20
            return super().get_dataset(x, y, setfit_args, max_pairs)

        # hard sample mining
        # TODO: something like this so we don't compute distances unless we need to?
        # if args.positive_sampling_strategy != "random" or args.negative_sampling_strategy != "random":
        #     embeddings = self.model.model_body.encode(
        #         x, normalize_embeddings=args.normalize_embeddings, convert_to_tensor=True
        #     )
        #     embeddings = embeddings.cpu()
        #     emb_dim = embeddings.shape[-1]

        #     distances = args.distance_metric(embeddings)
        # else:
        #     distances = []

        embeddings = self.model.model_body.encode(
            x, normalize_embeddings=args.normalize_embeddings, convert_to_tensor=True
        )
        embeddings = embeddings.cpu()
        emb_dim = embeddings.shape[-1]

        distances = args.distance_metric(embeddings)

        return_triplets = args.loss == TripletLoss
        # get pairs/triplets if triplet loss
        # TODO: maybe 'fixed' negative random pairs to evaluate the impact of positive pairs more seperately?
        pos_idx, neg_idx, rand_pos_idx, rand_neg_idx = get_samples(
            distances=distances,
            labels=y,
            args=args,
            return_triplets=return_triplets,
            emb_dim=emb_dim,
            rng=self.rng,
            n_rng=self.n_rng
        )

        all_pairs = []
        if return_triplets:
            for a, p, n in neg_idx:
                assert (
                    y[a] == y[p]
                ), "Samples in positive pair don't have the same label.. look at positive sample selection"
                assert (
                    y[a] != y[n]
                ), "Samples in negative pair have the same label.. look at negative sample selection"
                all_pairs.append({"anchor": x[a], "positive": x[p], "negative": x[n]})
        else:
            # extra shuffleing just because
            self.rng.shuffle(pos_idx)
            for pair in pos_idx:
                self.rng.shuffle(pair)
            self.rng.shuffle(neg_idx)
            for pair in neg_idx:
                self.rng.shuffle(pair)

            for p, q in pos_idx:
                assert (
                    y[p] == y[q]
                ), "Something went horribly wrong.. look at positive sample selection"
                all_pairs.append({"sentence_1": x[p], "sentence_2": x[q], "label": 1.0})
            for p, q in neg_idx:
                assert (
                    y[p] != y[q]
                ), "Something went horribly wrong.. look at negative sample selection"
                all_pairs.append({"sentence_1": x[p], "sentence_2": x[q], "label": 0.0})

        # shuffle and turn into dataset
        self.rng.shuffle(all_pairs)
        dataset = Dataset.from_list(all_pairs)
        if return_triplets:
            # need to use a different function for distance. TripletLoss expects different input (anchor,pos,neg)
            # compared to the 'Batch' versions of TripletLoss. 'Batch' version of distance function expects only one input
            # and computes pairwise distance for all sentences, while here we compute distances for positive and negative pairs seperately
            if args.distance_metric == BatchHardTripletLossDistanceFunction.eucledian_distance:
                triplet_distance_metric = TripletDistanceMetric.EUCLIDEAN
                # normalize if wanted (same as 'EUCLIDEAN' but normalizing embeddings before)
                if args.normalize_embeddings:
                    triplet_distance_metric = lambda x,y: F.pairwise_distance(F.normalize(x, p=2, dim=1), F.normalize(y, p=2, dim=1), p=2)
            else:
                triplet_distance_metric = TripletDistanceMetric.COSINE
                
            loss = args.loss(
                model=self.model.model_body,
                distance_metric=triplet_distance_metric,
                triplet_margin=args.margin,
            )
        else:
            loss = args.loss(self.model.model_body)

        if args.save_distances and not args.only_last_distance:
            self.distances.append(distances.tolist())
        if args.save_pairs:
            self.positive_pairs.append(pos_idx)
            self.negative_pairs.append(neg_idx)
            self.random_pos_pairs.append(rand_pos_idx)
            self.random_neg_pairs.append(rand_neg_idx)

        return dataset, loss

    def train_embeddings(
        self,
        x_train: list[str],
        y_train: list[int] | list[list[int]] | None = None,
        x_eval: list[str] | None = None,
        y_eval: list[int] | list[list[int] | None] = None,
        args: MiningArguments = None,
    ):
        args = args or self.args or MiningArguments()
        if args:
            self.st_trainer.setfit_args = args

        self.total_train_samples = 0
        sampling_times = []
        training_times = []

        print("bleh")
        logger.info("***** Running training *****")
        logger.info(f"  Num epochs = {args.embedding_num_epochs}")
        logger.info(f"  Batch size = {args.embedding_batch_size}")

        callbacks = list(self.st_trainer.callback_handler.callbacks)

        return_triplets = args.loss == TripletLoss

        if x_eval is not None: # and args.eval_strategy != IntervalStrategy.NO:
            # make eval args. basically just to set sampling strategy to unique
            # FIXME: Need to think of something for triplets here.. maybe just pick a big K
            if return_triplets:
                # just use the same for now?..
                # eval_args = args
                # alternatively use a lot of random samples? ,
                eval_args = MiningArguments(
                    sampling_strategy="random",
                    k=(10, 1),
                    loss=TripletLoss,
                    no_duplicates=True,
                    save_distances=False,
                    save_pairs=False,
                )
            else:
                eval_args = MiningArguments(sampling_strategy="unique")

        optim = None
        eval_bin_acc = {}
        log_hist = []
        for step in range(args.num_sampling_epochs):
            logger.info(f"  Current Sampling Step = {step}")

            sampling_start = time.perf_counter()
            train_dataset, loss = self.get_dataset(x_train, y_train, args=args)
            sampling_end = time.perf_counter() - sampling_start
            sampling_times.append(sampling_end)
            logger.info(f"  Sampling done in {sampling_end:.4f}s")
            len_train = len(train_dataset)
            if len_train == 0:
                logger.warning(f"  Found 0 training pairs. Stopping training!")
                break

            if x_eval is not None: # and args.eval_strategy != IntervalStrategy.NO:
                logger.debug(f"  Getting eval dataset")
                eval_dataset, _ = self.get_dataset(x_eval, y_eval, args=eval_args)
                if len(eval_dataset) == 0:
                    logger.warning(
                        f"  Sample mining returned an empty evaluation dataset!"
                    )
                    eval_dataset = None
            else:
                eval_dataset = None
            
            logger.info(f"  Num pairs = {len_train}")

            train_start = time.perf_counter()

            self.st_trainer = BCSentenceTransformersTrainer(
                setfit_model=self._model, setfit_args=args, callbacks=callbacks
            )

            num_batches = math.ceil(len_train / args.embedding_batch_size)
            est_train_steps = num_batches * args.embedding_num_epochs
            if not args.old_lr_schedule:
                est_train_steps = est_train_steps * args.num_sampling_epochs

            # overwrite warmup ratio if there are too few training steps
            if args.min_steps_for_warmup >= est_train_steps:
                logger.debug(
                    f"Not doing warmup as there are only {est_train_steps} steps."
                )
                self.st_trainer.args.warmup_ratio = 0.0

            # 'old' LR reset every sampling epoch.. but also decreased more aggressively.. a bit like the "warm resets" scheduler..
            # 'new' is just the default LR Scheduler being passed to the trainer every epoch.. (training could stop early cause no pairs fround..)
            if not args.old_lr_schedule and args.sampling_strategy == "mining":
                if optim is None:
                    logger.debug(
                        f"Creating LR Scheduler - Estimated total steps = {est_train_steps}"
                    )
                    self.st_trainer.create_optimizer_and_scheduler(
                        num_training_steps=est_train_steps
                    )
                    self.st_trainer._created_lr_scheduler = False
                    optim = (
                        self.st_trainer.optimizer,
                        self.st_trainer.lr_scheduler,
                    )
                else:
                    self.st_trainer.optimizer, self.st_trainer.lr_scheduler = optim
                    logger.debug(f"Last LR = {optim[1].get_last_lr()}")

            # train sentence transfomer!
            self.st_trainer.train_dataset = train_dataset
            self.st_trainer.eval_dataset = eval_dataset
            self.st_trainer.loss = loss

            self.st_trainer.train()
            train_stop = time.perf_counter() - train_start
            training_times.append(train_stop)
            self._model.model_body = self.st_trainer.model  # Probably don't need this.. should all reference the same model anyways..

            self.total_train_samples += len(train_dataset)
            log_hist.append(self.st_trainer.state.log_history)

            # NOTE: evaluate predicted similarity scores for eval dataset + get next eval dataset if not done
            if x_eval is not None:
                logger.info(f"  Evaluating on eval dataset")
                eval_embeddings = self.model.model_body.encode(
                    x_eval, normalize_embeddings=args.normalize_embeddings, batch_size=16, convert_to_tensor=True, show_progress_bar=True)

                eval_embeddings = eval_embeddings.detach().cpu()
                eval_stats = evaluate_pair_similarities(eval_embeddings, y_eval)
                eval_bin_acc[step] = eval_stats

            # FIXME DELETE!!!
            if self.save_embeddings:
                t_embs = self.model.model_body.encode(x_train)
                t_out = Path(self.args.output_dir).joinpath(f"train_emb_{step}.png")
                e_out = Path(self.args.output_dir).joinpath(f"eval_emb_{step}.png")
                plot_tsne(t_embs, y_train, self.train_dataset["label_text"], show=False, save_path=t_out)
                plot_tsne(eval_embeddings, y_eval, self.eval_dataset["label_text"], show=False, save_path=e_out)

        # saving stuff
        total_sampling_time = np.sum(sampling_times)
        total_training_time = np.sum(training_times)
        
        with open(Path(self.args.output_dir).joinpath("log_hist.json"), "w") as fp:
            json.dump(log_hist, fp)

        with open(Path(self.args.output_dir).joinpath("time.json"), "w") as fp:
            json.dump({"train_time": total_training_time, "sampling_time": total_sampling_time, "num_epochs": step}, fp)

        with open(Path(self.args.output_dir).joinpath("eval_stats.json"), "w") as fp:
            json.dump(eval_bin_acc, fp, indent=2)

        if self.args.save_pairs:
            pairs_path = Path(self.args.output_dir).joinpath("pairs.npz")
            positive_pairs = np.array(self.positive_pairs, dtype=object)
            negative_pairs = np.array(self.negative_pairs, dtype=object)
            random_positive_pairs = np.array(self.random_pos_pairs, dtype=object)
            random_negative_pairs = np.array(self.random_neg_pairs, dtype=object)

            np.savez(
                pairs_path,
                positive_pairs=positive_pairs,
                negative_pairs=negative_pairs,
                random_positives=random_positive_pairs,
                random_negatives=random_negative_pairs,
            )

        if self.args.save_distances:
            # get "last" distance after training,
            embs = self.model.model_body.encode(
                x_train,
                normalize_embeddings=self.args.normalize_embeddings,
                convert_to_tensor=True,
            )
            dist = self.args.distance_metric(embs)
            self.distances.append(dist.tolist())
            print_distance_stats(
                dist, y_train, margin=self.args.sampling_margin
            )
            if self.args.only_last_distance:
                self.distances = self.distances[-1]

            dist_path = Path(self.args.output_dir).joinpath("distances.npy")
            np.save(dist_path, self.distances)

        # 'delete' saved pairs/distances
        self.distances = self.positive_pairs = self.negative_pairs = []
        self.random_neg_pairs = self.random_pos_pairs = []

    def evaluate(
        self, dataset: Dataset | None = None, metric_key_prefix: str = "test"
    ) -> dict[str, float]:
        # modified https://github.com/huggingface/setfit/blob/main/src/setfit/trainer.py#L651
        # to show progress bar during evaluate
        """
        Computes the metrics for a given classifier.

        Args:
            dataset (`Dataset`, *optional*):
                The dataset to compute the metrics on. If not provided, will use the evaluation dataset passed via
                the `eval_dataset` argument at `Trainer` initialization.

        Returns:
            `Dict[str, float]`: The evaluation metrics.
        """

        if dataset is not None:
            self._validate_column_mapping(dataset)
            if self.column_mapping is not None:
                logger.info("Applying column mapping to the evaluation dataset")
                eval_dataset = self._apply_column_mapping(dataset, self.column_mapping)
            else:
                eval_dataset = dataset
        else:
            eval_dataset = self.eval_dataset

        if eval_dataset is None:
            raise ValueError(
                "No evaluation dataset provided to `Trainer.evaluate` nor the `Trainer` initialzation."
            )

        x_test = eval_dataset["text"]
        y_test = eval_dataset["label"]

        logger.info("***** Running evaluation *****")
        # modified this to be able to access the embeddings
        # y_pred = self.model.predict(x_test, use_labels=False)
        is_singular = isinstance(x_test, str)
        if is_singular:
            x_test = [x_test]
        embeddings = self.model.encode(x_test, batch_size=16, show_progress_bar=True)
        preds = self.model.model_head.predict(embeddings)
        y_pred = self.model._output_type_conversion(preds, as_numpy=False)
        y_pred = y_pred[0] if is_singular else y_pred

        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu()

        # Normalize string outputs
        if y_test and isinstance(y_test[0], str):
            encoder = LabelEncoder()
            encoder.fit(list(y_test) + list(y_pred))
            y_test = encoder.transform(y_test)
            y_pred = encoder.transform(y_pred)

        metric_kwargs = self.metric_kwargs or {}
        if isinstance(self.metric, str):
            metric_config = (
                "multilabel" if self.model.multi_target_strategy is not None else None
            )
            metric_fn = evaluate.load(self.metric, config_name=metric_config)

            results = metric_fn.compute(
                predictions=y_pred, references=y_test, **metric_kwargs
            )

        elif callable(self.metric):
            results = self.metric(y_pred, y_test, **metric_kwargs)

        else:
            raise ValueError("metric must be a string or a callable")

        if not isinstance(results, dict):
            results = {"metric": results}
        self.model.model_card_data.post_training_eval_results(
            {f"{metric_key_prefix}_{key}": value for key, value in results.items()}
        )

        # save test distances
        # distances = self.args.distance_metric(_convert_to_tensor(embeddings)).tolist()
        # just always use cosine distance for now
        # distances = BatchHardTripletLossDistanceFunction.cosine_distance(_convert_to_tensor(embeddings)).tolist()
        # dist_stats = get_distance_stats(distances, y_test, margin=self.args.sampling_margin)
        # if self.args.evaluate_test_dataset:
        #     dist_stats = evaluate_pair_similarities(embeddings, y_test)
        #     dist_path = Path(self.args.output_dir).joinpath("test-distances-stats.json")
        #     with open(dist_path, "w") as fp:
        #         json.dump(dist_stats, fp, indent=1)

        return results
