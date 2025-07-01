import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import os
import random
import logging

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.cnn import SimpleCNN
from strategies.random_sampling import random_sampling
from strategies.uncertainty_sampling import uncertainty_sampling
from strategies.bayesian_sampling import bayesian_sampling
from trainer.active_learner import active_learning_loop
from strategies.bayesian_diversity_sampling import bayesian_diversity_sampling
from strategies.uncertainty_diversity_sampling import uncertainty_diversity_sampling
from utils.plot_utils import plot_samples_needed, plot_accuracy_per_class_strategy

# --- Logging and Seed
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

set_all_seeds(42)

from collections import defaultdict

import numpy as np

def average_sample_results(all_results, strategies, classes):
    # Initialisiere Sammelstruktur für alle Samples je Strategie und Klasse
    avg_samples = {s: {c: [] for c in classes} for s in strategies.keys()}

    for result in all_results:
        for strategy, values in result.items():
            for (s_name, cls), sample_count in values.items():
                # Sicherheitshalber nur, wenn Strategie und Klasse bekannt sind
                if s_name in avg_samples and cls in avg_samples[s_name]:
                    avg_samples[s_name][cls].append(sample_count)

    # Mittelwert berechnen
    avg_samples_mean = {
        s: {c: np.mean(v) if v else 0 for c, v in cls_dict.items()}
        for s, cls_dict in avg_samples.items()
    }
    return avg_samples_mean


from collections import defaultdict

def average_accuracy_over_runs(all_acc_over_time_runs, all_classes, strategies):
    result = {strategy: {c: [] for c in all_classes} for strategy in strategies}

    for run_acc in all_acc_over_time_runs:
        for strategy, class_dict in run_acc.items():
            for c in all_classes:
                if isinstance(class_dict, dict):
                    acc_list = class_dict.get(c, [])
                else:
                    raise TypeError(f"Expected dict for class_dict, got {type(class_dict)}")
                result[strategy][c].append(acc_list)

    # Now compute average
    avg_result = {}
    for strategy in result:
        avg_result[strategy] = {}
        for c in result[strategy]:
            lists = result[strategy][c]
            if not lists:
                avg_result[strategy][c] = []
            else:
                max_len = max(len(l) for l in lists)
                padded = [l + [l[-1]] * (max_len - len(l)) if l else [0.0] * max_len for l in lists]
                avg = [sum(vals) / len(vals) for vals in zip(*padded)]
                avg_result[strategy][c] = avg

    return avg_result





# --- Config
QUERY_SIZE = 200
MAX_ITERATIONS = 50
INITIAL_CLASS = 0
THRESHOLD = 0.98
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Init wandb
wandb.login(key="5509e4778375d7a603e946cae346f3a939027ed9")

# --- Load Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_train = datasets.MNIST(root="data", train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root="data", train=False, download=True, transform=transform)

# --- Define strategies
strategies = {
    "Random": random_sampling,
    "Uncertainty": uncertainty_sampling,
    "bayesian": bayesian_sampling,
    "uncertainty_diversity": uncertainty_diversity_sampling,
    "bayesian_diversity": bayesian_diversity_sampling
}

NUM_RUNS = 3
all_classes = list(range(10))

# Neue Strukturen
all_final_results_runs = []
all_acc_over_time_runs = []

for run_idx in range(NUM_RUNS):
    logger.info(f"\n\n🚀 Starting Run {run_idx + 1}/{NUM_RUNS}")
    set_all_seeds(42 + run_idx)
    class_order = random.sample(all_classes, len(all_classes))

    final_results, acc_over_time = active_learning_loop(
        project_name="mnist_active_learning",
        strategies=strategies,
        model_class=SimpleCNN,
        dataset=mnist_train,
        test_dataset=mnist_test,
        class_order=class_order,
        device=DEVICE,
        query_size=QUERY_SIZE,
        max_iterations=MAX_ITERATIONS,
        initial_class=class_order[0],
        threshold=THRESHOLD
    )

    all_final_results_runs.append(final_results)       # {strategy_name: {(strategy, class): count}}
    all_acc_over_time_runs.append(acc_over_time)       # {strategy_name: {class: [acc, acc, ...]}}



avg_acc_per_strategy = average_accuracy_over_runs(all_acc_over_time_runs, all_classes, strategies)


avg_samples_mean = average_sample_results(all_final_results_runs, strategies, all_classes)


# --- Plotting Results
os.makedirs("results", exist_ok=True)
plot_samples_needed(avg_samples_mean, output_path="thesis/results/mnist/samples_needed_avg.png")
plot_accuracy_per_class_strategy(avg_acc_per_strategy)

