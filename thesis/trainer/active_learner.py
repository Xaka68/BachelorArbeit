import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import logging
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from collections import defaultdict
from metrics.accuracy import get_class_accuracy

logger = logging.getLogger(__name__)

def train(model, dataloader, optimizer, criterion, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

def active_learning_loop(
    project_name,
    model_class,
    dataset,
    test_dataset,
    strategies,
    class_order,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    query_size=200,
    max_iterations=50,
    threshold=0.99,
    initial_class=None
):
    final_results = {}
    acc_over_time_all = {}  # <-- FIXED: Store per-strategy accuracy over time

    all_classes = sorted(set([label for _, label in dataset]))
    logger.info(f"Detected {len(all_classes)} classes: {all_classes}")

    if initial_class is None:
        initial_class = class_order[0]

    for strategy_name, strategy_fn in strategies.items():
        wandb.login()
        run = wandb.init(project=project_name, name=strategy_name, reinit=True)
        wandb.config.update({
            "query_size": query_size,
            "max_iterations": max_iterations,
            "threshold": threshold,
            "class_order": class_order
        })

        included_classes = [initial_class]
        targets_np = np.array([label for _, label in dataset])
        initial_class_indices = np.where(targets_np == initial_class)[0]
        sample_size = min(258, len(initial_class_indices))

        labeled_indices = np.random.choice(initial_class_indices, size=sample_size, replace=False).tolist()
        full_unlabeled_pool = list(set(range(len(dataset))) - set(labeled_indices))

        labeled_counts = {i: 0 for i in all_classes}
        for idx in labeled_indices:
            labeled_counts[dataset[idx][1]] += 1

        sample_needed = {}
        acc_over_time = {i: [] for i in all_classes}
        model = model_class().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        for cls in class_order:
            if cls == initial_class:
                continue

            logger.info(f"\n🔁 Adding class {cls}")
            included_classes.append(cls)

            # Reset unlabeled pool
            unlabeled_indices = full_unlabeled_pool.copy()

            # 🟡 Remember count before learning this class
            labeled_count_before = len(labeled_indices)

            for iteration in range(1, max_iterations + 1):
                logger.info(f"→ Iteration {iteration} with classes {included_classes}")

                labeled_loader = DataLoader(Subset(dataset, labeled_indices), batch_size=64, shuffle=True)
                test_filtered = [(x, y) for x, y in test_dataset if y in included_classes]
                test_loader = DataLoader(test_filtered, batch_size=256, shuffle=False)

                train(model, labeled_loader, optimizer, criterion, device=device, epochs=10)

                accs = get_class_accuracy(model, test_loader, device)
                for c, a in accs.items():
                    acc_over_time[c].append(a)

                wandb.log({f"accuracy_class_{k}": v for k, v in accs.items()})

                if accs.get(cls, 0.0) >= threshold:
                    newly_labeled = len(labeled_indices) - labeled_count_before
                    logger.info(f"✅ Class {cls} reached {threshold*100:.0f}% accuracy with {newly_labeled} new labeled samples.")
                    sample_needed[(strategy_name, cls)] = newly_labeled
                    break

                # Use full unlabeled pool
                x_pool_all = [(dataset[i][0], dataset[i][1], i) for i in unlabeled_indices]
                if not x_pool_all:
                    logger.info("⚠️ Pool exhausted.")
                    break

                x_pool = torch.stack([x for x, _, _ in x_pool_all])
                dataset_indices = [i for _, _, i in x_pool_all]

                if strategy_name in ["uncertainty_diversity", "bayesian_diversity"]:
                    x_labeled = torch.stack([dataset[i][0] for i in labeled_indices]) if labeled_indices else None
                    indices = strategy_fn(model, x_labeled, x_pool, query_size)
                else:
                    indices = strategy_fn(model, x_pool, query_size)

                mapped_indices = [dataset_indices[i] for i in indices]
                for idx in mapped_indices:
                    labeled_counts[dataset[idx][1]] += 1

                labeled_indices.extend(mapped_indices)
                unlabeled_indices = list(set(unlabeled_indices) - set(mapped_indices))

        final_results[strategy_name] = sample_needed
        acc_over_time_all[strategy_name] = acc_over_time  # <-- FIXED: Store all strategies' accuracies
        run.finish()

    return final_results, acc_over_time_all  # <-- FIXED: Return correct structure
