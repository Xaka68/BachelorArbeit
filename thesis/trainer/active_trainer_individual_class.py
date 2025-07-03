import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import wandb
import random

from torch.utils.data import DataLoader, Subset
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


def class_wise_active_learning_loop(
    project_name,
    model_class,
    dataset,
    test_dataset,
    strategies,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    query_size=200,
    max_iterations=100
):
    results = {}  # strategy -> class -> (percentage queried, accuracy)

    all_classes = sorted(set(label for _, _,label in dataset))
    for strategy_name, strategy_fn in strategies.items():
        logger.info(f"\n=== Running strategy: {strategy_name} ===")
        wandb.init(project=project_name, name=strategy_name, reinit=True, settings=wandb.Settings(init_timeout=1200))
        results[strategy_name] = {}

        for cls in sorted(random.sample(all_classes, 10)):
            logger.info(f"\n--- Training model for class {cls} ---")

            all_indices = list(range(len(dataset)))
            labeled_indices = []
            unlabeled_indices = all_indices.copy()

            acc_progress = []

            while unlabeled_indices:
                model = model_class().to(device)
                optimizer = optim.Adam(model.parameters(), lr=1e-3)
                criterion = nn.CrossEntropyLoss()

                if labeled_indices:
                    labeled_subset = Subset(dataset, labeled_indices)
                    labeled_loader = DataLoader(labeled_subset, batch_size=64, shuffle=True)
                    train(model, labeled_loader, optimizer, criterion, device, epochs=10)

                # Evaluate on class-specific test data
                test_class_data = [(x, y) for x, y in test_dataset if y == cls]
                print(f"Evaluating on {len(test_class_data)} examples for class {cls}")
                test_loader = DataLoader(test_class_data, batch_size=256, shuffle=False)
                acc = get_class_accuracy(model, test_loader, device).get(cls, 0.0)

                print(f"Class {cls}, Accuracy: {acc:.4f}")

                percent = len(labeled_indices) / len(dataset)
                acc_progress.append((percent, acc))

                wandb.log({
                    f"accuracy/class_{cls}": acc,
                    f"percent_queried/class_{cls}": percent
                })

                if not unlabeled_indices:
                    break

                # Build pool of all unlabeled examples
                x_pool_all = torch.stack([dataset[i][0] for i in unlabeled_indices])

                if strategy_name == "uncertainty_diversity" or strategy_name == "bayesian_diversity":
                    x_labeled = torch.stack([dataset[i][0] for i in labeled_indices]) if labeled_indices else None
                    selected_relative_indices = strategy_fn(model, x_labeled, x_pool_all, query_size)
                else:
                    selected_relative_indices = strategy_fn(model, x_pool_all, query_size)

                # Map selected indices back to original dataset indices
                selected_indices = [unlabeled_indices[i] for i in selected_relative_indices]
                print(f"Selected indices for class {cls}: {selected_indices}")

                labeled_indices.extend(selected_indices)
                unlabeled_indices = list(set(unlabeled_indices) - set(selected_indices))

            results[strategy_name][cls] = acc_progress

        wandb.finish()

    return results
