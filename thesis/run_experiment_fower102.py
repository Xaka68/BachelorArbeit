import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import wandb
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
from models.resnet_wrapper import ResNetWrapper
from strategies.random_sampling import random_sampling
from strategies.uncertainty_sampling import uncertainty_sampling
from strategies.bayesian_sampling import bayesian_sampling
from trainer.active_learner import active_learning_loop
from utils.plot_utils import plot_samples_needed, plot_accuracy_over_time


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

torch.cuda.empty_cache()


from torchvision.datasets import Flowers102
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import ConcatDataset

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # adjust to model input size
    transforms.ToTensor(),
])


class Flower102Dataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.data = Flowers102(root=root, split=split, download=True, transform=transform)
        self.targets = self.data._labels

    def __getitem__(self, idx):
        image, label = self.data[idx]  # already transformed
        return image, label

    def __len__(self):
        return len(self.data)



# --- Config
query_size = 25
max_iterations = 50
initial_class = 0
threshold = 0.90

# --- Load Dataset
dataset = Flower102Dataset(root="./data", split="train", transform=transform)
#val_dataset = Flower102Dataset(root="./data", split="val", transform=transform)
test_dataset = Flower102Dataset(root="./data", split="test", transform=transform)
#combined = ConcatDataset([test_dataset, val_dataset])
#test_dataset = combined  # Use combined dataset for testing

# --- Strategy Map
strategies = {
    "Random": random_sampling,
    "Uncertainty": uncertainty_sampling,
}

# --- Run Active Learning
final_results, acc_over_time = active_learning_loop(
    project_name="flower102_active_learning",
    model_class=ResNetWrapper,
    dataset=dataset,
    test_dataset=test_dataset,
    strategies=strategies,
    query_size=query_size,
    max_iterations=max_iterations,
    threshold=threshold,
    initial_class=initial_class,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# --- Plotting Results
os.makedirs("results", exist_ok=True)
plot_samples_needed(final_results, output_path="thesis/results/flower102/samples_needed_per_class.png")
plot_accuracy_over_time(acc_over_time, output_path="thesis/results/flower102/accuracy_over_time.png")

# --- Plot: Samples Needed Per Class
plt.figure(figsize=(10, 6))
for strat, cls_dict in final_results.items():
    cls_ids = sorted(cls_dict.keys(), key=lambda x: x[1])
    x = [cls for (_, cls) in cls_ids]
    y = [cls_dict[(strat, cls)] for (_, cls) in cls_ids]
    plt.plot(x, y, marker='o', label=strat)

plt.xlabel("New Class ID")
plt.ylabel("Labeled Samples Needed to Reach 99% Accuracy")
plt.title("Comparison of Query Strategies on Class Detection (Flower102)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("samples_needed_per_class_flower102.png")
plt.show()

# --- Plot: Accuracy Over Time
for cls in range(102):
    if cls in acc_over_time and acc_over_time[cls]:
        plt.plot(acc_over_time[cls], label=f"Class {cls}")

plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.title("Class Accuracy over Time (Flower102)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("accuracy_over_time_flower102.png")
plt.show()