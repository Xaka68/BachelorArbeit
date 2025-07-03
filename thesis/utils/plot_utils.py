import matplotlib.pyplot as plt
import os

def plot_samples_needed(final_results, output_path="samples_needed_per_class.png"):
    plt.figure(figsize=(10, 6))
    for strat, cls_dict in final_results.items():
        cls_ids = sorted(cls_dict.keys())  # keys sind jetzt ints (class IDs)
        x = cls_ids
        y = [cls_dict[cls] for cls in cls_ids]
        plt.plot(x, y, marker='o', label=strat)

    plt.xlabel("New Class ID")
    plt.ylabel("Labeled Samples Needed to Reach Threshold Accuracy")
    plt.title("Comparison of Query Strategies on Class Detection")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



def plot_accuracy_over_time(acc_over_time, output_path="accuracy_over_time.png"):
    plt.figure(figsize=(10, 6))
    for cls in range(10):
        if cls in acc_over_time and acc_over_time[cls]:
            plt.plot(acc_over_time[cls], label=f"Class {cls}")

    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Class Accuracy over Time")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


import os
import matplotlib.pyplot as plt

def plot_accuracy_per_class_strategy(avg_acc_per_strategy, output_dir="results/mnist/acc_curves"):
    os.makedirs(output_dir, exist_ok=True)

    # Get the union of all classes across strategies
    all_classes = sorted({c for accs in avg_acc_per_strategy.values() for c in accs})

    for cls in all_classes:
        plt.figure(figsize=(8, 5))
        for strategy in sorted(avg_acc_per_strategy.keys()):
            accs = avg_acc_per_strategy[strategy]
            if cls in accs:
                plt.plot(accs[cls], label=strategy)
        plt.title(f"Class {cls} Accuracy over Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.0)
        plt.legend(title="Strategy")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"class_{str(cls).zfill(2)}_acc.png"))
        plt.close()





import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d  # for smoothing

def plot_class_accuracy_vs_data_percentage(
    results,
    output_dir="thesis/results/mnist/all_strategies",
    smooth=True,
    window_size=3,
    y_min=0.9,
    y_max=1.0
):
    os.makedirs(output_dir, exist_ok=True)
    for cls in range(10):
        plt.figure(figsize=(10, 6))
        for strategy_name, strategy_data in results.items():
            if cls not in strategy_data:
                continue

            # Sort data points by percentage
            data_points = sorted(strategy_data[cls], key=lambda x: x[0])
            percents, accs = zip(*data_points)
            percents = np.array(percents) * 100  # convert to percentage scale
            accs = np.array(accs)

            # Smooth accuracy curve if needed
            if smooth and len(accs) >= window_size:
                accs_smooth = uniform_filter1d(accs, size=window_size)
            else:
                accs_smooth = accs

            plt.plot(
                percents,
                accs_smooth,
                label=strategy_name,
                marker='o',
                markersize=6,
                linewidth=2
            )

        plt.title(f"Class {cls}: Accuracy vs % of Queried Data (Zoomed 0.9 - 1.0)")
        plt.xlabel("% of Queried Data")
        plt.ylabel("Accuracy")
        plt.ylim(y_min, y_max)  # zoom y-axis to 0.9 - 1.0
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"class_{cls}_accuracy_curve_zoomed.png"))
        plt.close()



# utils/plot_utils.py
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d  # for smoothing

def plot_class_accuracy_vs_data_percentage_2(
    results,
    output_dir="thesis/results/mnist/class_accuracy_charts",
    smooth=True,
    window_size=3,
    y_min=0.9,
    y_max=1.0
):
    os.makedirs(output_dir, exist_ok=True)

    # Get all unique classes from results
    all_classes = set()
    for strategy_data in results.values():
        all_classes.update(strategy_data.keys())

    for cls in sorted(all_classes):
        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()

        # Offset the third axis
        ax3.spines["right"].set_position(("axes", 1.1))

        for strategy_name, strategy_data in results.items():
            if cls not in strategy_data:
                continue

            data_points = sorted(strategy_data[cls], key=lambda x: x["percent_queried"])
            percents = np.array([dp["percent_queried"] for dp in data_points]) * 100
            accs = np.array([dp["accuracy"] for dp in data_points])
            cls_percents = np.array([dp["percent_selected_class"] for dp in data_points]) * 100
            entropies = np.array([dp["avg_entropy"] for dp in data_points])
            l2_norms = np.array([dp["model_l2_norm"] for dp in data_points])

            # Smooth
            if smooth and len(accs) >= window_size:
                accs = uniform_filter1d(accs, size=window_size)
                cls_percents = uniform_filter1d(cls_percents, size=window_size)
                entropies = uniform_filter1d(entropies, size=window_size)
                l2_norms = uniform_filter1d(l2_norms, size=window_size)

            ax1.plot(percents, accs, label=f"Acc - {strategy_name}", marker='o')
            ax2.plot(percents, cls_percents, label=f"% Class - {strategy_name}", linestyle='--')
            ax3.plot(percents, entropies, label=f"Entropy - {strategy_name}", linestyle=':')
            ax3.plot(percents, l2_norms, label=f"L2 Norm - {strategy_name}", linestyle='-.')

        ax1.set_xlabel("% of Queried Data")
        ax1.set_ylabel("Accuracy")
        ax1.set_ylim(y_min, y_max)
        ax2.set_ylabel("% Newly Queried from Class")
        ax3.set_ylabel("Avg Entropy / L2 Norm")

        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_title(f"Class {cls}: Accuracy, Class Query %, Entropy, and L2 Norm vs Queried Data")

        # Combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        lines_3, labels_3 = ax3.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2 + lines_3, labels_1 + labels_2 + labels_3)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"class_{cls}_multi_metric_curve.png"))
        plt.close(fig)



def plot_class_accuracy_vs_data_percentage_3(
    results,
    output_dir="thesis/results/mnist/entropy_diverse/class_accuracy_charts",
    smooth=True,
    window_size=3,
    y_min=0.9,
    y_max=1.0
):
    os.makedirs(output_dir, exist_ok=True)

    # Get all unique classes from results
    all_classes = set()
    for strategy_data in results.values():
        all_classes.update(strategy_data.keys())

    for cls in sorted(all_classes):
        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()

        # Offset the third axis
        ax3.spines["right"].set_position(("axes", 1.1))

        for strategy_name, strategy_data in results.items():
            if cls not in strategy_data:
                continue

            data_points = sorted(strategy_data[cls], key=lambda x: x["percent_queried"])
            percents = np.array([dp["percent_queried"] for dp in data_points]) * 100
            accs = np.array([dp["accuracy"] for dp in data_points])
            cls_percents = np.array([dp["percent_selected_class"] for dp in data_points]) * 100
            entropies = np.array([dp["avg_entropy"] for dp in data_points])
            l2_norms = np.array([dp["model_l2_norm"] for dp in data_points])

            # Smooth
            if smooth and len(accs) >= window_size:
                accs = uniform_filter1d(accs, size=window_size)
                cls_percents = uniform_filter1d(cls_percents, size=window_size)
                entropies = uniform_filter1d(entropies, size=window_size)
                l2_norms = uniform_filter1d(l2_norms, size=window_size)

            ax1.plot(percents, accs, label=f"Acc - {strategy_name}", marker='o')
            ax2.plot(percents, cls_percents, label=f"% Class - {strategy_name}", linestyle='--')
            ax3.plot(percents, entropies, label=f"Entropy - {strategy_name}", linestyle=':')
            ax3.plot(percents, l2_norms, label=f"L2 Norm - {strategy_name}", linestyle='-.')

        ax1.set_xlabel("% of Queried Data")
        ax1.set_ylabel("Accuracy")
        ax1.set_ylim(y_min, y_max)
        ax2.set_ylabel("% Newly Queried from Class")
        ax3.set_ylabel("Avg Entropy / L2 Norm")

        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_title(f"Class {cls}: Accuracy, Class Query %, Entropy, and L2 Norm vs Queried Data")

        # Combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        lines_3, labels_3 = ax3.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2 + lines_3, labels_1 + labels_2 + labels_3)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"class_{cls}_multi_metric_curve.png"))
        plt.close(fig)



def plot_class_accuracy_vs_data_percentage_4(
    results,
    output_dir="thesis/results/mnist/diverse_entropy/class_accuracy_charts",
    smooth=True,
    window_size=3,
    y_min=0.9,
    y_max=1.0
):
    os.makedirs(output_dir, exist_ok=True)

    # Get all unique classes from results
    all_classes = set()
    for strategy_data in results.values():
        all_classes.update(strategy_data.keys())

    for cls in sorted(all_classes):
        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()

        # Offset the third axis
        ax3.spines["right"].set_position(("axes", 1.1))

        for strategy_name, strategy_data in results.items():
            if cls not in strategy_data:
                continue

            data_points = sorted(strategy_data[cls], key=lambda x: x["percent_queried"])
            percents = np.array([dp["percent_queried"] for dp in data_points]) * 100
            accs = np.array([dp["accuracy"] for dp in data_points])
            cls_percents = np.array([dp["percent_selected_class"] for dp in data_points]) * 100
            entropies = np.array([dp["avg_entropy"] for dp in data_points])
            l2_norms = np.array([dp["model_l2_norm"] for dp in data_points])

            # Smooth
            if smooth and len(accs) >= window_size:
                accs = uniform_filter1d(accs, size=window_size)
                cls_percents = uniform_filter1d(cls_percents, size=window_size)
                entropies = uniform_filter1d(entropies, size=window_size)
                l2_norms = uniform_filter1d(l2_norms, size=window_size)

            ax1.plot(percents, accs, label=f"Acc - {strategy_name}", marker='o')
            ax2.plot(percents, cls_percents, label=f"% Class - {strategy_name}", linestyle='--')
            ax3.plot(percents, entropies, label=f"Entropy - {strategy_name}", linestyle=':')
            ax3.plot(percents, l2_norms, label=f"L2 Norm - {strategy_name}", linestyle='-.')

        ax1.set_xlabel("% of Queried Data")
        ax1.set_ylabel("Accuracy")
        ax1.set_ylim(y_min, y_max)
        ax2.set_ylabel("% Newly Queried from Class")
        ax3.set_ylabel("Avg Entropy / L2 Norm")

        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_title(f"Class {cls}: Accuracy, Class Query %, Entropy, and L2 Norm vs Queried Data")

        # Combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        lines_3, labels_3 = ax3.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2 + lines_3, labels_1 + labels_2 + labels_3)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"class_{cls}_multi_metric_curve.png"))
        plt.close(fig)