# import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import json
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd
from util import (
    load_datasets,
    SAMPLE_SIZES,
    DEV_DATASET_TO_METRIC,
    TEST_DATASET_TO_METRIC,
    print_distance_stats,
)
from sentence_transformers.losses import BatchHardTripletLoss

import matplotlib.pyplot as plt

DATASETS = list(DEV_DATASET_TO_METRIC) + list(TEST_DATASET_TO_METRIC)

tableau20 = [
    (31, 119, 180),
    (174, 199, 232),
    (255, 127, 14),
    (255, 187, 120),
    (44, 160, 44),
    (152, 223, 138),
    (214, 39, 40),
    (255, 152, 150),
    (148, 103, 189),
    (197, 176, 213),
    (140, 86, 75),
    (196, 156, 148),
    (227, 119, 194),
    (247, 182, 210),
    (127, 127, 127),
    (199, 199, 199),
    (188, 189, 34),
    (219, 219, 141),
    (23, 190, 207),
    (158, 218, 229),
]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255.0, g / 255.0, b / 255.0)
    # np.random.shuffle(tableau20)

tableau20 = tableau20[::2]

# NOTE: kind of nothing interesting to see here because results are so close (except failure cases like topk)
def plot_metrics(results, names, sample_sizes=None):
    num_results = len(results)
    datasets = list(results[0].keys())
    print(datasets)
    if sample_sizes is None:
        sample_sizes = list(results[0][datasets[0]].keys())

    bar_width = 0.8 / num_results
    x = np.arange(len(datasets))

    reverse = False
    for s in sample_sizes:
        s = str(s)  # in case we input sample sizes
        # get mean values and std
        mean_results = []
        std_results = []
        for r in results:
            r_means = {}
            r_std = {}
            for ds in datasets:
                if isinstance(r[ds][s][0], int):
                    reverse = True
                mult = 1 if reverse else 100
                mean = np.mean(r[ds][s]) * mult
                std = np.std(r[ds][s]) * mult
                r_means[ds] = mean
                r_std[ds] = std
            # r_means = {ds: np.mean(x) * 100 for ds in datasets for x in r[ds][s]}
            # r_std = {ds: np.std(x) * 100 for ds in datasets for x in r[ds][s]}
            mean_results.append(r_means)
            std_results.append(r_std)

        best_indices = []
        for ds in datasets:
            means = [m[ds] for m in mean_results]
            if reverse:
                best = np.argmin(means)
            else:
                best = np.argmax(means)
            best_indices.append(best)

        fig = plt.figure(figsize=(12, 6))
        # colors = plt.get_cmap("tab10", num_results)
        colors = plt.get_cmap("tab10", num_results)

        for i in range(len(mean_results)):
            ds_means = [mean_results[i][ds] for ds in datasets]
            ds_std = [std_results[i][ds] for ds in datasets]
            offset = (i - (num_results - 1) / 2) * bar_width

            bar_colors = [colors(i)]
            # bar_colors = tableau20[i]
            # hatches = ["--" if best_indices[j] == i else "" for j in range(len(datasets))] 
            bars = plt.bar(
                x + offset,
                ds_means,
                bar_width,
                yerr=ds_std,
                capsize=7,
                label=names[i],
                color=bar_colors,
                edgecolor="black",
                linewidth=1.0,
                # hatch=hatches,
            )

            for j, bar in enumerate(bars):
                height = bar.get_height() 
                plt.text(bar.get_x() + bar.get_width()/2, height + ds_std[j],
                        #  "${{{:.2f}}}_{{{:.2f}}}$".format(round(height,2), round(ds_std[j], 2)),
                         "${{{:.2f}}}$".format(round(height,2)),
                         ha="center", va="bottom", fontsize=8,
                         c = "red" if best_indices[j] == i else "black",
                        #  usetex=True
                         )

        short_ds_names = [d[:10] for d in datasets]
        plt.xticks(x, short_ds_names, rotation="vertical")
        plt.title(f"Performance for sample size {s}")
        plt.xlabel("Dataset")
        plt.ylabel("Metric %")
        plt.legend()
        # get rid of hatches on plots..
        ax = plt.gca()
        leg = ax.get_legend()
        for i in range(len(leg.legend_handles)):
            # leg.legend_handles[i].set_color(tableau20[i])
            leg.legend_handles[i].set_color(colors(i))

        ax.spines[["right", "top"]].set_visible(False)
        plt.tight_layout()
        plt.show()

