import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import wandb
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset
from metrics.accuracy import get_class_accuracy

logger = logging.getLogger(__name__)

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

def compute_entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-10), dim=1)

def compute_model_l2_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.requires_grad:
            total_norm += p.norm(2).item() ** 2
    return total_norm ** 0.5

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
    results = {}  # strategy -> class -> list of dicts with metrics

    all_classes = sorted(set(label for _, label in dataset))
    for strategy_name, strategy_fn in strategies.items():
        logger.info(f"\n=== Running strategy: {strategy_name} ===")
        wandb.init(project=project_name, name=strategy_name, reinit=True)
        results[strategy_name] = {}

        for cls in all_classes:
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
                    train(model, labeled_loader, optimizer, criterion, device)

                # Evaluate on class-specific test data
                test_class_data = [(x, y) for x, y in test_dataset if y == cls]
                test_loader = DataLoader(test_class_data, batch_size=256, shuffle=False)
                acc = get_class_accuracy(model, test_loader, device).get(cls, 0.0)

                print(f"Class {cls}, Accuracy: {acc:.4f}")

                # Percent labeled from entire dataset
                percent = len(labeled_indices) / len(dataset)

                # Placeholder for additional metrics
                percent_selected_cls = None
                avg_entropy = None

                if not unlabeled_indices:
                    break

                # Build pool of all unlabeled examples
                x_pool_all = torch.stack([dataset[i][0] for i in unlabeled_indices])

                if strategy_name == "Hybrid":
                    x_labeled = torch.stack([dataset[i][0] for i in labeled_indices]) if labeled_indices else None
                    selected_relative_indices = strategy_fn(model, x_labeled, x_pool_all, query_size)
                else:
                    selected_relative_indices = strategy_fn(model, x_pool_all, query_size)

                # Map selected indices back to original dataset indices
                selected_indices = [unlabeled_indices[i] for i in selected_relative_indices]

                # Count how many of the newly selected samples are from the training class
                selected_cls_count = sum(1 for i in selected_indices if dataset[i][1] == cls)
                percent_selected_cls = selected_cls_count / len(selected_indices) if selected_indices else 0.0

                # Compute average entropy for newly selected samples (Uncertainty Sampling)
                if strategy_name == "Uncertainty Sampling" and selected_indices:
                    model.eval()
                    with torch.no_grad():
                        logits = model(torch.stack([dataset[i][0].to(device) for i in selected_indices]))
                        probs = F.softmax(logits, dim=1)
                        entropies = compute_entropy(probs)
                        avg_entropy = entropies.mean().item()

                model_l2_norm = compute_model_l2_norm(model)

                wandb.log({
                    f"accuracy/class_{cls}": acc,
                    f"percent_queried/class_{cls}": percent,
                    f"percent_selected_class_{cls}": percent_selected_cls,
                    f"avg_entropy_selected_class_{cls}": avg_entropy,
                    f"model_l2_norm/class_{cls}": model_l2_norm
                })

                acc_progress.append({
                    "percent_queried": percent,
                    "accuracy": acc,
                    "percent_selected_class": percent_selected_cls,
                    "avg_entropy": avg_entropy,
                    "model_l2_norm": model_l2_norm
                })

                labeled_indices.extend(selected_indices)
                unlabeled_indices = list(set(unlabeled_indices) - set(selected_indices))

            results[strategy_name][cls] = acc_progress

        wandb.finish()

    return results
