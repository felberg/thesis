import json
import time
import argparse
import logging
from pathlib import Path

from collections import defaultdict
from typing import Any, Callable

import torch
from torch.multiprocessing import Process, Queue

from datasets import Dataset
from trainer import MiningTrainer
from train_args import MiningArguments

from util import (
    load_datasets,
    load_validation_dataset,
    load_model,
    SAMPLE_SIZES,
    ALL_DATASETS_TO_METRIC,
    DATASET_TO_NUM_LABELS,
    LOSS_MAP,
    DISTANCE_FUNCTION_MAP,
)
from metrics_util import get_full_metrics

logging.basicConfig(level=logging.WARN)
logging.getLogger("trainer").setLevel(logging.DEBUG)
logging.getLogger("sampler").setLevel(logging.DEBUG)
logging.getLogger("util").setLevel(logging.DEBUG)
logging.getLogger("transformers.trainer_callback").setLevel(
    logging.CRITICAL
)  # NOTE: supresses a warning about adding a callback when already present.. maybe look into a different way to fix this.


FULL_METRICS = False
EVAL = False

def parse_args():
    args = argparse.ArgumentParser("SetFit-Mining")
    args.add_argument(
        "--model",
        default="paraphrase-mpnet-base-v2",
        help="Base model to use with SetFit",
    )
    args.add_argument(
        "--datasets", nargs="+", default=["sst2"], help="list of datasets"
    )
    args.add_argument(
        "--sample_sizes",
        type=int,
        nargs="+",
        default=SAMPLE_SIZES,
        help="list of sample sizes",
    )
    args.add_argument("--output_dir", default="./results")
    args.add_argument("--cache_dir", default="./cache_dir")
    args.add_argument("--config", default=None)
    args.add_argument("--run_name", default=None)
    args.add_argument(
        "--num_iterations",
        type=int,
        default=None,
        help="Set this to 20 for 'legacy' setfit",
    )
    args.add_argument("--sampling_strategy", default="random-semi")
    args.add_argument("--sampling_steps", type=int, default=1)
    args.add_argument("--loss", default="CosineSimilarityLoss")
    args.add_argument(
        "--distance_metric", default="cosine", choices=["cosine", "euclidean"]
    )
    args.add_argument(
        "--margin",
        type=float,
        default=0.2,
        help="Margin value used for triplet loss. Determines how far apart negative pairs should be from positive pairs",
    )
    args.add_argument(
        "--sampling_margin",
        type=float,
        default=0.2,
        help="Margin value to determine (semi-)hard sample pairs. ",
    )
    args.add_argument(
        "--k", type=int, default=1, help="number of pairs to draw for each sample"
    )
    args.add_argument(
        "--k_n",
        type=int,
        default=None,
        help="number of negative pairs to draw for each positive pair",
    )
    args.add_argument(
        "--skip_k", type=int, default=0, help="number of pairs to oversample and skip when using topk"
    )
    args.add_argument(
        "--max_p",
        type=int,
        default=0,
        help="max number of positive pairs",
    )
    args.add_argument(
        "--max_n",
        type=int,
        default=0,
        help="max number of negative pairs",
    )
    args.add_argument(
        "--min_p",
        type=int,
        default=0,
        help="min number of positive pairs",
    )
    args.add_argument(
        "--min_n",
        type=int,
        default=0,
        help="min number of negative pairs",
    )
    args.add_argument(
        "--low_p",
        type=float,
        default=None,
        help="either fixed lower distance bound for positive pairs or a 'quantile' of the closest positives that gets masked",
    )
    args.add_argument(
        "--low_n",
        type=float,
        default=None,
        help="either fixed lower distance bound for negatives pairs or a 'quantile' of the closest negatives that gets masked",
    )
    args.add_argument(
        "--high_p",
        type=float,
        default=None,
    )
    args.add_argument(
        "--high_n",
        type=float,
        default=None,
    )
    args.add_argument("--old_lr", default=False, action="store_true")
    args.add_argument("--no_loop", default=False, action="store_true")
    args.add_argument("--quantile_limits", default=False, action="store_true")
    args.add_argument("--duplicates", default=False, action="store_true")
    args.add_argument("--independent_negatives", default=False, action="store_true")
    args.add_argument("--remove_pos_pairs", default=False, action="store_true")
    args.add_argument("--use_pair_dist", default=False, action="store_true")
    args.add_argument("--normalize_embeddings", default=False, action="store_true")
    args.add_argument("--all_datasets", default=False, action="store_true")
    args.add_argument("--fast_datasets", default=False, action="store_true")
    args.add_argument("--overwrite", default=False, action="store_true")
    args.add_argument("--disable_saving", default=False, action="store_true")
    args.add_argument("--spawn_subprocess", default=False, action="store_true")
    return args.parse_args()


def train_func(
    model_name: str,
    train_args: MiningArguments,
    train_dataset: Dataset | None = None,
    test_dataset: Dataset | None = None,
    eval_dataset: Dataset | None = None,
    metric: str | Callable = "accuracy",
    metric_kwargs: dict[str, Any] | None = None,
    cache_dir: str | None = None,
    queue=None,
):
    model = load_model(model_name, cache_dir=cache_dir)

    # got a convergence warning on some datasets.. didn't seem to impact performance but increasing iterations, as suggested in the warning..
    model_head_params = model.model_head.get_params()
    model_head_params["max_iter"] = 1000
    model.model_head.set_params(**model_head_params)

    callbacks = []

    trainer = MiningTrainer(
        model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        metric=metric,
        metric_kwargs=metric_kwargs,
        callbacks=callbacks,
    )
    trainer.train()

    eval_start = time.perf_counter()
    metrics = trainer.evaluate(test_dataset)
    print(f"Evaluation finished in {time.perf_counter() - eval_start} s!")

    num_samples = trainer.total_train_samples
    metrics["total_samples"] = num_samples

    if queue is not None:
        queue.put(metrics)
    else:
        return metrics
    
def argparse_to_mining_args(args):
    if args.config is not None:
        print("Loading Mining Arguments from " + args.config)
        config_path = Path(args.config).expanduser()
        with open(config_path, "r") as f:
            d = json.load(f)
        return MiningArguments.from_json_dict(d)
    else:
        k = args.k if args.k_n is None else (args.k, args.k_n)
        return MiningArguments(
            batch_size=(16, 2),
            num_epochs=(1, 16),
            sampling_strategy=args.sampling_strategy,
            num_sampling_epochs=args.sampling_steps,
            num_iterations=args.num_iterations,
            loss=LOSS_MAP[args.loss],
            distance_metric=DISTANCE_FUNCTION_MAP[args.distance_metric],
            margin=args.margin,
            sampling_margin=args.sampling_margin,
            k=k,
            skip_k=args.skip_k,
            max_pairs=(args.max_p, args.max_n),
            min_pairs=(args.min_p, args.min_n),
            run_name=args.run_name,
            max_length=256,
            logging_steps=10,
            eval_strategy="no",
            save_strategy="no",
            pos_lim=(args.low_p, args.high_p),
            neg_lim=(args.low_n, args.high_n),
            no_duplicates=not args.duplicates,
            independent_negatives=args.independent_negatives,
            use_pair_dist=args.use_pair_dist,
            normalize_embeddings=args.normalize_embeddings,
            old_lr_schedule=args.old_lr,
            no_loop=args.no_loop,
            quantile_limits=args.quantile_limits,
            remove_pos_pairs=args.remove_pos_pairs,
            save_pairs=not args.disable_saving,
            save_distances=not args.disable_saving,
            evaluate_test_dataset=not args.disable_saving,
            only_last_distance=False,
            report_to="none"
        )

if __name__ == "__main__":
    print("------ SetFitMining ------")
    args = parse_args()
    # make output directory
    if args.output_dir:
        main_dir = Path(args.output_dir).expanduser()
    else:
        main_dir = Path.cwd().joinpath("results")

    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else None

    datasets = args.datasets
    if args.all_datasets:
        datasets = list(ALL_DATASETS_TO_METRIC)
    elif args.fast_datasets:
        datasets = ["sst2", "sst5", "emotion", "amazon_counterfactual_en"]

    print(f"Datasets: {datasets}")

    # create mining args just to get method name to save stuff
    m_args = argparse_to_mining_args(args)
    method_name = m_args.get_run_name()

    output_dir = main_dir / method_name
    output_dir.mkdir(parents=True,exist_ok=True)

    # save mining parameters as well, why not
    mining_args_path = output_dir / "mining_args.json"
    with open(mining_args_path, "w") as f:
        json.dump(m_args.to_json_dict(), f, indent=2)

    result_file_name = "results.json"
    all_results_file = output_dir / result_file_name
    all_results = {}

    if all_results_file.exists():
        # loading previously accumulated results
        with open(all_results_file, "r") as f:
            all_results = json.load(f)

    # setup datasets and metrics
    datasets = args.datasets
    if args.all_datasets:
        datasets = list(ALL_DATASETS_TO_METRIC)

    # training
    for d in datasets:
        dataset_results = {}
        dataset_dir = output_dir / d
        dataset_dir.mkdir(parents=True,exist_ok=True)
        dataset_results_path = dataset_dir / result_file_name

        train_splits, test_dataset = load_datasets(d, sample_sizes=args.sample_sizes, cache_dir=cache_dir)
        eval_dataset = None

        num_unique_labels = len(set(train_splits[f"train-{args.sample_sizes[0]}-0"]["label"]))
        print(f"{d} - {num_unique_labels} classes")
        # metric
        metric_kwargs = None
        metric = "accuracy"
        if d in ALL_DATASETS_TO_METRIC:
            metric = ALL_DATASETS_TO_METRIC[d]

        if FULL_METRICS:
            combined_metric, metric_kwargs = get_full_metrics(metric, num_unique_labels)
        else:
            combined_metric = metric

        for split_name, train_dataset in train_splits.items():
            cur_sample_size = split_name.split("-")[1]
            cur_split = split_name.split("-")[2]
            split_dir = dataset_dir.joinpath(cur_sample_size + "-samples", cur_split)
            split_dir.mkdir(parents=True, exist_ok=True)
            split_result_file = split_dir / result_file_name

            if cur_sample_size not in dataset_results:
                dataset_results[cur_sample_size] = defaultdict(list)

            print(
                f"------ Running training on {d} - {cur_sample_size} samples - split {cur_split} ------"
            )
            print(f"saving to {split_result_file}")

            # skip if already done
            if split_result_file.exists() and not args.overwrite:
                print("Already done! Skipping.")
                # load results so we can append to overarching dictionaries
                with open(split_result_file) as f:
                    old_results = json.load(f)
                    for k, v in old_results.items():
                        dataset_results[cur_sample_size][k].append(v)

                continue

            train_args = argparse_to_mining_args(args)
            train_args.output_dir = split_dir

            if EVAL and eval_dataset:
                train_args.eval_strategy = "steps"
                train_args.eval_steps = train_args.logging_steps * 10 

            # just running training in a loops caused my machine to go OOM.
            # -> just spawn a subprocess..
            if args.spawn_subprocess:
                queue = Queue()
                process = Process(
                    target=train_func,
                    args=(
                        args.model,
                        train_args,
                        train_dataset,
                        test_dataset,
                        eval_dataset,
                        combined_metric,
                        metric_kwargs,
                        cache_dir,
                        queue,
                    ),
                )
                process.start()
                process.join()
                # should be fine since join already blocks and this way if an exception occured -> queue empty -> we don't block forever on get()
                metrics = queue.get(False)
            else:
                print(f"CUDA MEMORY ALLOC BEFORE TRAINING: {torch.cuda.memory_allocated()}")
                metrics = train_func(
                    model_name=args.model,
                    train_args=train_args,
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    eval_dataset=eval_dataset,
                    metric=combined_metric,
                    metric_kwargs=metric_kwargs,
                    cache_dir=cache_dir,
                )

            #print(f"CUDA MEMORY ALLOC AFTER TRAINING: {torch.cuda.memory_allocated()}")
            for k, v in metrics.items():
                dataset_results[cur_sample_size][k].append(v)

            with open(split_result_file, "w") as f:
                json.dump(metrics, f)

        with open(dataset_results_path, "w") as f:
            json.dump(dataset_results, f)

        all_results[d] = dataset_results

    with open(all_results_file, "w") as f:
        json.dump(all_results, f, indent=2)

