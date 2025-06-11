from enum import Enum
import os
import pickle
import json
import logging
from typing import Any
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch import Tensor
import evaluate
from sklearn.metrics import average_precision_score, matthews_corrcoef
from sklearn.manifold import TSNE
from scipy import stats
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers.losses import (
    BatchHardTripletLoss,
    BatchHardTripletLossDistanceFunction,
    TripletLoss,
)
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from setfit import SetFitModel, Trainer
from setfit.utils import (
    load_data_splits,
    TEST_DATASET_TO_METRIC,
    DEV_DATASET_TO_METRIC,
    LOSS_NAME_TO_CLASS,
)
from setfit.data import SAMPLE_SIZES, sample_dataset

from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

### DEFAULTS
ALL_DATASETS_TO_METRIC = TEST_DATASET_TO_METRIC | DEV_DATASET_TO_METRIC

DISTANCE_FUNCTION_MAP = {
    "cosine": BatchHardTripletLossDistanceFunction.cosine_distance,
    "euclidean": BatchHardTripletLossDistanceFunction.eucledian_distance,  # NOTE: nice typo?
}
DISTANCE_FUNCTION_MAP.update(dict([reversed(i) for i in DISTANCE_FUNCTION_MAP.items()]))
# add TripletLoss to LOSS_NAME_TO_CLASS
LOSS_NAME_TO_CLASS["TripletLoss"] = TripletLoss
LOSS_MAP = LOSS_NAME_TO_CLASS | dict([reversed(i) for i in LOSS_NAME_TO_CLASS.items()])

DATASET_TO_NUM_LABELS = {
    "sst2": 2,
    "imdb": 2,
    "subj": 2,
    "bbc-news": 5,
    "student-question-categories": 4,
    "TREC-QC": 50,
    "toxic_conversations": 2,
    "emotion": 6,
    "SentEval-CR": 2,
    "sst5": 5,
    "ag_news": 4,
    "enron_spam": 2,
    "amazon_counterfactual_en": 2,
    "amazon_polarity": 2,
}


def _convert_to_tensor(x: list | np.ndarray | Tensor) -> Tensor:
    if isinstance(x, list):
        # convert to numpy first in rare case of list of ndarrays
        x = np.array(x)
    if not isinstance(x, Tensor):
        x = torch.tensor(x)
    return x


def _convert_to_numpy(x: list | np.ndarray | Tensor) -> np.ndarray:

    if isinstance(x, list):
        x = np.array(x)
    if isinstance(x, Tensor):
        x = x.cpu().numpy()
    return x


def _get_fake_labels(sample_size: int, num_labels: int) -> np.ndarray:
    return np.array([i for i in range(num_labels)]).repeat(sample_size)


def get_labels(
    dataset: str,
    sample_size: int,
    split: int,
    cache_dir: str | os.PathLike | None = None,
) -> list:
    # load dataset split and get labels
    ds = load_datasets(dataset, sample_sizes=[sample_size], cache_dir=cache_dir)[0][
        f"train-{sample_size}-{split}"
    ]
    labels = ds["label"]
    return labels


def load_validation_dataset(
    dataset: str,
    samples_per_class: int = 100,  # NOTE: maybe better to just draw a fixed number e.g. 1000 samples for every dataset?..
    cache_dir: str | os.PathLike | None = None,
) -> Dataset:
    # try loading
    ds_name = "validation-" + str(samples_per_class)
    if cache_dir:
        logger.info(f"Trying to load validation dataset from '{cache_dir}'")
        dataset_dir = Path(cache_dir, dataset).expanduser()
        ds_path = dataset_dir / ds_name
        if ds_path.exists():
            return Dataset.load_from_disk(ds_path)

    # haven't saved validation set before
    full_ds = load_dataset(f"SetFit/{dataset}", split="train")
    validation_split = sample_dataset(full_ds, num_samples=samples_per_class)

    if cache_dir:
        # saving
        logger.info(f"Saving validation dataset to '{cache_dir}'")
        save_path = Path(cache_dir, dataset, ds_name).expanduser()
        validation_split.save_to_disk(str(save_path))

    return validation_split


def load_datasets(
    dataset: str,
    sample_sizes: list[int] = SAMPLE_SIZES,
    cache_dir: str | os.PathLike | None = None,
) -> tuple[DatasetDict, Dataset]:
    # try loading
    if cache_dir:
        dataset_dir = Path(cache_dir, dataset).expanduser()
        json_fname = "dataset_dict.json"
        json_path = dataset_dir / json_fname
        if dataset_dir.exists() and json_path.is_file():
            print(f"Loading dataset '{dataset}' from {cache_dir}.")

            with open(json_path, "r") as f:
                split_names = json.load(f)["splits"]
                split_names.remove("test")

            missing = []
            for s in sample_sizes:
                if f"train-{s}-0" not in split_names:
                    missing.append(s)

            data_dict = DatasetDict()
            if missing:
                print(f"Downloading missing sample sizes: {missing}")
                train_splits, _ = load_data_splits(
                    dataset=dataset, sample_sizes=missing, add_data_augmentation=False
                )

                print(f"Saving missing datasets to {cache_dir}")
                missing_splits = [*train_splits]
                split_names = sorted(
                    split_names + missing_splits, key=lambda s: int(s.split("-")[1])
                )
                for key, ds in train_splits.items():
                    ds.save_to_disk(str(dataset_dir.joinpath(key)))
                with open(json_path, "w") as f:
                    json.dump({"splits": split_names + ["test"]}, f)

            for s in split_names:
                _s = int(s.split("-")[1])
                if _s in sample_sizes:
                    if _s in missing:
                        data_dict[s] = train_splits[s]
                    else:
                        # print(s.split("-")[1])
                        data_dict[s] = Dataset.load_from_disk(
                            str(dataset_dir.joinpath(s))
                        )

            test_dataset = Dataset.load_from_disk(str(dataset_dir.joinpath("test")))

            return data_dict, test_dataset

    # else load from hub
    train_splits, test_dataset = load_data_splits(
        dataset=dataset, sample_sizes=sample_sizes, add_data_augmentation=False
    )

    if dataset_dir:
        # save to local if saving enabled
        print(f"Saving dataset '{dataset}' to {cache_dir}")
        data_dict_copy = DatasetDict(train_splits)
        data_dict_copy["test"] = test_dataset
        data_dict_copy.save_to_disk(str(dataset_dir))

    return train_splits, test_dataset

def load_model(
    model_name: str, cache_dir: str | os.PathLike | None = None
) -> SetFitModel:

    if cache_dir:
        model_dir = Path(cache_dir, model_name).expanduser()
        if model_dir.exists():
            try:
                print(f"Trying to load model from {model_dir}...")
                model = SetFitModel.from_pretrained(model_dir, local_files_only=True)
                return model
            except Exception as e:
                print(f"Couldn't load model from local files.. {e.args}")

    print("Loading model from huggingface...")
    model = SetFitModel.from_pretrained(model_name)
    if cache_dir:
        print(f"Saving model to {model_dir}")
        model.save_pretrained(model_dir)

    return model


def get_mask(
    labels: list | np.ndarray | Tensor,
    positive: bool,
    target_label: int | None = None,
) -> Tensor:

    mask_fun = (
        BatchHardTripletLoss.get_anchor_positive_triplet_mask
        if positive
        else BatchHardTripletLoss.get_anchor_negative_triplet_mask
    )

    labels = _convert_to_tensor(labels)

    mask = mask_fun(labels)

    # keep only pairs where a sample belongs to our target label
    if target_label is not None:
        label_mask = labels == target_label
        mask = mask * label_mask
        # make symmetrical
        mask = mask ^ mask.T

    return mask


# NOTE: had to do this like this, because using the BinaryClassificationEvaluator
# caused massive memory issues, probably due to doubly embedding the sentences..
def evaluate_pair_similarities(embeddings, labels) -> dict:
    # get distances
    _convert_to_tensor(embeddings)
    logger.debug("Calculating distances...")
    distances = BatchHardTripletLossDistanceFunction.cosine_distance(
        embeddings
    ).tolist()
    logger.debug("Calculating stats...")
    distance_stats = get_distance_stats(distances, labels)
    # get all pairs
    p_mask = get_mask(labels, positive=True)
    n_mask = get_mask(labels, positive=False)
    all_pos_idx = torch.nonzero(torch.triu(p_mask)).tolist()
    all_neg_idx = torch.nonzero(torch.triu(n_mask)).tolist()
    predictions = []
    references = []
    for p, n in all_pos_idx:
        predictions.append(1 - distances[p][n])
        references.append(1.0)
    for p, n in all_neg_idx:
        predictions.append(1 - distances[p][n])
        references.append(0.0)

    predictions = np.array(predictions)
    references = np.array(references)
    # make dummy evaluator
    evaluator = BinaryClassificationEvaluator([""], [""], [1])
    # basically a copy of BinaryClassificationEvaluator.compute_metrices()
    acc, acc_threshold = evaluator.find_best_acc_and_threshold(
        predictions, references, True
    )
    logger.info(f"Accuracy:  {acc * 100:.2f}\t(Threshold: {acc_threshold:.4f})")

    f1, precision, recall, f1_threshold = evaluator.find_best_f1_and_threshold(
        predictions, references, True
    )
    logger.info(f"F1:        {f1 * 100:.2f}\t(Threshold: {f1_threshold:.4f})")
    logger.info(f"Precision: {precision * 100:.2f}")
    logger.info(f"Recall:    {recall * 100:.2f}")

    ap = average_precision_score(references, predictions)
    logger.info(f"Average:   {ap * 100:.2f}")

    predicted_labels = predictions >= f1_threshold
    mcc = matthews_corrcoef(references, predicted_labels)
    logger.info(f"Matthews:  {mcc * 100:.2f}\n")

    out = {
        "accuracy": acc,
        "accuracy_threshold": acc_threshold,
        "f1": f1,
        "f1_threshold": f1_threshold,
        "precision": precision,
        "recall": recall,
        "ap": ap,
        "mcc": mcc,
        "distance_stats": distance_stats,
    }
    return out


# NOTE: maybe split this into like results_util.py or something?
class SplitResultFile(Enum):
    # RESULTS = "results.json"
    DISTANCES = "distances.npy"
    PAIRS = "pairs.npz"
    TIME = "time.json"
    TEST_DISTANCES = "test-distances-stats.json"
    EVAL_CSV = "binary_classification_evaluation_eval_results.csv"  # NOTE: ...maybe change the save name? this is a bit too long..


def load_split_result(
    main_path: str | os.PathLike,
    dataset: str,
    sample_size: int,
    split: int,
    target_file: SplitResultFile,
) -> list[np.ndarray] | dict | pd.DataFrame:
    main_dir = Path(main_path).expanduser()
    file = (
        main_dir / dataset / f"{sample_size}-samples" / str(split) / target_file.value
    )
    if not file.is_file():
        raise FileNotFoundError(f"File not found at '{file}'")

    if target_file == SplitResultFile.DISTANCES:
        return np.load(file)
    elif target_file == SplitResultFile.PAIRS:
        return np.load(file, allow_pickle=True)
    elif target_file == SplitResultFile.TIME:
        with open(file, "r") as f:
            return json.load(f)
    elif target_file == SplitResultFile.TEST_DISTANCES:
        with open(file, "r") as f:
            return json.load(f)
    elif target_file == SplitResultFile.EVAL_CSV:
        return pd.read_csv(file)
    # elif target_file == SplitResultFile.RESULTS:
    #     with open(file, "r") as f:
    #         return json.load(f)


def load_results(
    paths: str | os.PathLike | list[str] | list[os.PathLike],
    datasets: list[str] = list(ALL_DATASETS_TO_METRIC),
    sample_sizes: list[int] = SAMPLE_SIZES,
    metric: str | None = None,
):
    # expect results file to be in path or full path to file
    results_dict = {}
    no_files = []
    # handle single path input
    if not isinstance(paths, list):
        paths = [paths]

    for p_index, p in enumerate(paths):
        method_results = {}
        p_path = Path(p).expanduser()
        if p_path.is_file() and p_path.suffix == ".json":
            results_file = p_path
        else:
            results_file = p_path / "results.json"
        if not results_file.is_file():
            # raise FileNotFoundError(f"Results file not found at '{results_file}'")
            print(f"Results file not found at '{results_file}'")
            no_files.append(p_index)
            continue
        with open(results_file, "r") as f:
            results = json.load(f)

        for d in datasets:
            # NOTE: maybe load everything if metric is None?
            if d in results:
                if results[d] == {}:
                    method_results[d] = None
                    continue

                method_results[d] = {}
                for s in sample_sizes:
                    if str(s) in results[d]:
                        print("test")
                        print(results[d])
                    
                        try: 
                            method_results[d][str(s)] = results[d][str(s)]
                        except KeyError:
                            print(f"KeyError: path - {p} -- ({d} , {s})")
                            method_results[d][str(s)] = None
                    else:
                        method_results[d][str(s)] = None
                        
               
            else:
                method_results[d] = None

        results_dict[p_index] = method_results

    if no_files:
        for i in reversed(no_files):
            del paths[i]
    return results_dict


# FIXME very bad
def print_comparison_table(
    results: dict,
    paths: list[str] | None = None,
    datasets: list[str] | None = None,
    sample_sizes: list[int] | None = None,
    sort_by: str | None = None,
    metric: str | None = None,
    show_pairs: bool = True,
):
    method_names = (
        [
            str(path).replace("\\", "/").split("/")[-1].removesuffix("-results.json")
            for path in paths
        ]
        if paths
        else [f"Method {i}" for i in results.keys()]
    )
    first_res = next(iter(results))
    print(first_res)
    print(results.keys())
    if datasets is None:
        datasets = list(results[first_res].keys())
        print(datasets)
    if sample_sizes is None:
        sample_sizes = SAMPLE_SIZES

    # print(datasets)
    # print(sample_sizes)
    # print(method_names)
    print("Comparison table")
    all_dfs = []
    for s in sample_sizes:
        print("#" * 15 + f" Sample size: {s} " + "#" * 15 + "\n")
        s_means = []
        s_stds = []
        s_both = []
        for method_key, method_results in results.items():
            m_means = []
            m_stds = []
            m_both = []
            for d in datasets:
                if d in method_results.keys():
                    if metric is None:
                        if d in ALL_DATASETS_TO_METRIC:
                            m = ALL_DATASETS_TO_METRIC[d]
                        else:
                            m = "accuracy"
                    else:
                        m = metric if metric in method_results[d][str(s)] else "accuracy"

                    vals = [0]
                    num_pairs = [0]
                    pairs_metric = "total_samples"
                    if method_results[d]:
                        if method_results[d][str(s)]:
                            vals = method_results[d][str(s)][m]
                            num_pairs = method_results[d][str(s)][pairs_metric]

                    mean = (
                        np.mean(vals) * 100
                        if not isinstance(vals[0], int)
                        else np.mean(vals)
                    )
                    std = (
                        np.std(vals) * 100
                        if not isinstance(vals[0], int)
                        else np.std(vals)
                    )
                    both = (
                        f"{mean:.2f} ± {std:.2f}"
                        if not isinstance(vals[0], int)
                        else f"{mean:.2f}"
                    )
                    if show_pairs:
                        both += " x " + str(int(np.mean(num_pairs)))
                   
                    m_means.append(mean)
                    m_stds.append(std)
                    m_both.append(both)

            s_means.append(m_means)
            s_stds.append(m_stds)
            s_both.append(m_both)

        # put into pd dataframe for easy printing
        ds_column_names = [d[:14] for d in datasets] + ["avg"]
        avg_mean = [np.mean(s) for s in s_means]
        avg_std = [np.mean(s) for s in s_stds]
        avg_both = [f"{mean:.2f} ± {std:.2f}" for mean, std in zip(avg_mean, avg_std)]
        if metric == "total_samples":
            avg_both = [f"{mean:.2f}" for mean in avg_mean]

        for i in range(len(s_both)):
            s_both[i].append(avg_both[i])
            s_means[i].append(avg_mean[i])

        df = pd.DataFrame(s_both, index=method_names, columns=ds_column_names)
        if sort_by:
            mean_df = pd.DataFrame(s_means, index=method_names, columns=ds_column_names)
            # sorted_index = mean_df.sort_values(by=sort_by[:14], ascending=False).index
            sort_by = sort_by[:14]

            sorted_index = mean_df.sort_values(by=sort_by, ascending=False).index
            df = df.loc[sorted_index]
        print(df.head(20))
        print()
        all_dfs.append(df)

    return all_dfs


def get_distance_stats(
    distances: list | np.ndarray | Tensor,
    labels: list | np.ndarray | Tensor,
    margin: float | None = None,
) -> dict:
    stats = {}
    distances = _convert_to_numpy(distances)
    labels = _convert_to_tensor(labels)

    mask_pos = get_mask(labels, positive=True).numpy()
    mask_neg = get_mask(labels, positive=False).numpy()
    num_pos = np.count_nonzero(mask_pos)
    num_neg = np.count_nonzero(mask_neg)
    stats["count"] = [num_pos, num_neg]

    dist_pos = distances * mask_pos
    dist_neg = np.where(mask_neg, distances, np.inf)
    # print(dist_pos)
    hardest_pos = np.max(dist_pos, axis=-1, keepdims=True)
    # print(pos_max)
    hardest_neg = np.min(dist_neg, axis=-1, keepdims=True)

    num_hard_pos = np.count_nonzero(dist_pos > hardest_neg)
    num_hard_neg = np.count_nonzero(dist_neg < hardest_pos)
    stats["hard"] = [num_hard_pos, num_hard_neg]
    if margin:
        num_semi_pos = np.count_nonzero(
            dist_pos > np.clip(hardest_neg - margin, a_min=0, a_max=None)
        )
        num_semi_neg = np.count_nonzero(dist_neg < (hardest_pos + margin))
        stats["semi"] = [num_semi_pos, num_semi_neg]

    dist_pos = distances[mask_pos]
    dist_neg = distances[mask_neg]

    stats["mean"] = [np.mean(dist_pos), np.mean(dist_neg)]
    stats["std"] = [np.std(dist_pos), np.std(dist_neg)]
    stats["min"] = [np.min(dist_pos), np.min(dist_neg)]
    stats["25%"] = [np.quantile(dist_pos, q=0.25), np.quantile(dist_neg, q=0.25)]
    stats["50%"] = [np.quantile(dist_pos, q=0.50), np.quantile(dist_neg, q=0.50)]
    stats["75%"] = [np.quantile(dist_pos, q=0.75), np.quantile(dist_neg, q=0.75)]
    stats["max"] = [np.max(dist_pos), np.max(dist_neg)]

    return stats


def print_distance_stats(
    distances: list | np.ndarray | Tensor,
    labels: list | np.ndarray | Tensor | int,
    margin: float | None = None,
    names: list[str] | None = None,
):
    distances = _convert_to_numpy(distances)
    labels = _convert_to_tensor(labels)

    num_samples = distances.shape[-1]
    if isinstance(labels, int):
        # fake labels assuming data was in order
        labels = _get_fake_labels(num_samples, labels)

    if len(distances.shape) < 3:
        distances = np.expand_dims(distances, axis=0)
        # print(distances.shape)

    num_distances = len(distances)
    all_dict = defaultdict(list)
    for i, d in enumerate(distances):
        all_dict[" "].append(f"{i}-Pos.")
        all_dict[" "].append(f"{i}-Neg.")
        stats = get_distance_stats(d, labels, margin)
        for key, value in stats.items():
            for v in value:
                all_dict[key].append(v)

    # print(all_dict)
    if names:
        for i, n in enumerate(names):
            print(f"{i:>2}: {n:<}")
        print()
    row_format = "{:<6}|" + "{:>12}{:>12} |" * num_distances
    for key, value in all_dict.items():
        row = row_format.format(
            str(key).upper(),
            *[
                f"{float(v):.5f}" if isinstance(v, (float, np.float32)) else f"{v}"
                for v in value
            ],
        )
        print(row)
        if key == " ":
            print("-" * len(row))


def pair_stuff(pairs, labels, distances):
    pos_pairs = pairs["positive_pairs"].tolist()
    neg_pairs = pairs["negative_pairs"].tolist()
    # rand_pos_pairs = pairs["random_positives"]
    # rand_neg_pairs = pairs["random_negatives"]

    labels = _convert_to_numpy(labels)
    num_labels = len(np.unique(labels))
    labels = labels.tolist()
    print("Number unique labels: ", num_labels)

    num_iter = len(pos_pairs)
    # print(rand_pos_pairs.shape)
    # print(rand_neg_pairs.shape)
    # print(rand_pos_pairs)

    # print(pos_pairs)
    test_pos = Counter([tuple(x) for xs in pos_pairs for x in xs])
    print("Most common pairs across all training iterations:")
    print("Positive: ", test_pos.most_common(5))
    test_neg = Counter([tuple(x) for xs in neg_pairs for x in xs])
    print("Negative: ", test_neg.most_common(5))
    pos_dupes = [v for k, v in test_pos.items() if v > 1]
    neg_dupes = [v for k, v in test_neg.items() if v > 1]
    print(
        f"Total number of positive duplicate pairs across iterations: {len(pos_dupes)} | {sum(pos_dupes)} (total: {test_pos.total()})"
    )
    print(
        f"Total number of negative duplicate pairs across iterations: {len(neg_dupes)} | {sum(neg_dupes)} (total: {test_neg.total()})"
    )

    print()

    for i in range(num_iter):
        pos_p = pos_pairs[i]
        neg_p = neg_pairs[i]
        d = distances[i]

        num_pos = len(pos_p)
        num_neg = len(neg_p)
        pos_dist = []
        pos_labels = []
        max_pos = ([], 0)
        min_pos = ([], 100)
        for p, q in pos_p:
            pos_labels.append(labels[p])
            assert labels[p] == labels[q], "erm"
            max_pos = ([p, q], d[p][q]) if d[p][q] > max_pos[1] else max_pos
            min_pos = ([p, q], d[p][q]) if d[p][q] < min_pos[1] else min_pos
            pos_dist.append(d[p][q])

        neg_dist = []
        neg_labels = []
        max_neg = ([], 0)
        min_neg = ([], 100)
        for p, q in neg_p:
            neg_labels.append(tuple(sorted([labels[p], labels[q]])))
            assert labels[p] != labels[q], "erm"
            max_neg = ([p, q], d[p][q]) if d[p][q] > max_neg[1] else max_neg
            min_neg = ([p, q], d[p][q]) if d[p][q] < min_neg[1] else min_neg
            neg_dist.append(d[p][q])

        print(f"Pos: {num_pos}  |  Neg: {num_neg}")
        print(
            f"Pos. Dist.: {np.mean(pos_dist):.4f}  |  Max: {max_pos[0]}, {float(max_pos[1]):.4f}  | Min: {min_pos[0]}, {float(min_pos[1]):.4f}"
        )
        print(
            f"neg. Dist.: {np.mean(neg_dist):.4f}  |  Max: {max_neg[0]}, {float(max_neg[1]):.4f}  | Min: {min_neg[0]}, {float(min_neg[1]):.4f}"
        )
        c_pos_labels = Counter(pos_labels)
        c_neg_labels = Counter(neg_labels)
        c_nn_labels = Counter([x for xs in neg_labels for x in xs])
        print(
            c_pos_labels.most_common(None),
            " ",
            len(c_pos_labels.keys()),
            "/",
            c_pos_labels.total(),
        )
        print(
            c_neg_labels.most_common(5),
            " ",
            len(c_neg_labels.keys()),
            "/",
            c_neg_labels.total(),
        )
        print(
            c_nn_labels.most_common(5),
            " ",
            len(c_nn_labels.keys()),
            "/",
            c_nn_labels.total(),
        )
        print()


def plot_tsne(
    embeddings, labels, label_text=None, perplexity=30.0, show=True, save_path=None
):
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(
        embeddings
    )

    plt.figure(figsize=(8, 6))
    uni_labels = np.unique(labels)
    uni_text = np.unique(label_text)
    print(uni_text)

    for i, cls in enumerate(uni_labels):
        idxs = labels == cls
        plt.scatter(tsne[idxs, 0], tsne[idxs, 1], label=uni_text[i], alpha=0.6, s=20)

    plt.legend(title="Class", loc="upper left")
    plt.tight_layout()
    if save_path:
        plt.savefig(Path(save_path))
    if show:
        plt.show()

    plt.close()


if __name__ == "__main__":
    import argparse
    from train_args import MiningArguments

    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=str, nargs="+", default=None)
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=list(ALL_DATASETS_TO_METRIC)
    )
    parser.add_argument("--sample_sizes", type=int, nargs="+", default=SAMPLE_SIZES)
    parser.add_argument("--metric", type=str, default=None)
    parser.add_argument("--sort_by", type=str, default=None)
    parser.add_argument("--show_pairs", default=False, action="store_true")
    args = parser.parse_args()

    if args.paths:
        results = load_results(
            args.paths, args.datasets, args.sample_sizes, args.metric
        )
        print_comparison_table(results, args.paths)
        
