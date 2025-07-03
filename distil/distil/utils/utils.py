import torch
from torch.utils.data import Dataset

class SubsetWithTargets(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices, labels):
        self.dataset = torch.utils.data.Subset(dataset, indices)
        self.targets = labels.type(torch.long)
    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)

class SubsetWithTargetsSingleChannel(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices, labels):
        self.dataset = torch.utils.data.Subset(dataset, indices)
        self.targets = labels.type(torch.long)
    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        image = torch.repeat_interleave(image, 3, 0)
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)


class ConcatWithTargets(Dataset):
    r"""
    Concat of a dataset at specified indices.
    """
    def __init__(self, dataset1, dataset2):
        self.dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])
        self.targets = torch.Tensor(list(dataset1.targets) + list(dataset2.targets)).type(torch.long)
    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)


class LabeledToUnlabeledDataset(Dataset):
    r"""
    Remove labels from a labeled dataset.
    """
    def __init__(self, wrapped_dataset):
        self.wrapped_dataset = wrapped_dataset

    def __getitem__(self, index):
        item = self.wrapped_dataset[index]
        if isinstance(item, dict) and 'labels' in item:
            item = {k: v for k, v in item.items() if k != 'labels'}
        return item

    def __len__(self):
        return len(self.wrapped_dataset)
    



import os
import numpy as np
from sklearn import metrics
import csv

def evaluate_class_performance(acc_curve, samples_per_round):
    n_rounds = len(acc_curve)
    x_axis = np.arange(n_rounds) * samples_per_round
    auc = metrics.auc(x_axis, acc_curve)

    final_acc = acc_curve[-1]
    target_acc = 0.9 * final_acc
    round_reached = np.argmax(acc_curve >= target_acc)
    samples_needed = round_reached * samples_per_round
    total_samples = samples_per_round * (n_rounds - 1)
    label_saving = 1 - (samples_needed / total_samples) if total_samples > 0 else 0.0
    return auc, label_saving

def evaluate_strategy_results(strategy_dir, nclasses, samples_per_round):
    strategy_results = {}
    for class_id in range(nclasses):
        file_path = os.path.join(strategy_dir, f'class_{class_id}_FASS.txt')
        if not os.path.exists(file_path):
            continue
        with open(file_path, 'r') as f:
            acc_curve = np.array([float(line.strip()) for line in f])
            auc, label_saving = evaluate_class_performance(acc_curve, samples_per_round)
            strategy_results[class_id] = {'AUC': auc, 'LabelSaving': label_saving}
    return strategy_results

def save_results_to_csv(results_dict, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['Class', 'AUC', 'LabelSaving']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for class_id, metrics in results_dict.items():
            writer.writerow({'Class': class_id, 'AUC': metrics['AUC'], 'LabelSaving': metrics['LabelSaving']})
