import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import os
import random
import logging

from torchvision import transforms
from torchvision.datasets import Flowers102
from torch.utils.data import DataLoader, Subset

from models.cnn import SimpleCNN
from strategies.random_sampling import random_sampling
from strategies.uncertainty_sampling import uncertainty_sampling
from strategies.uncertainty_diversity_sampling import uncertainty_diversity_sampling
from strategies.bayesian_sampling import bayesian_sampling
from strategies.bayesian_diversity_sampling import bayesian_diversity_sampling
from trainer.active_trainer_individual_class import class_wise_active_learning_loop
from utils.plot_utils import plot_class_accuracy_vs_data_percentage

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

# --- Config
QUERY_SIZE = 50
MAX_ITERATIONS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Init wandb
wandb.login(key="5509e4778375d7a603e946cae346f3a939027ed9")

# --- Load Flower102 Dataset
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

flower_train = Flowers102(root="data", split="train", download=True, transform=transform)
flower_test = Flowers102(root="data", split="test", download=True, transform=transform)


print(f" training dataset size: {len(flower_train)}")
print(f" test dataset size: {len(flower_test)}")

# --- Define Strategies
strategies = {
    "uncertainty_diversity": uncertainty_diversity_sampling
}


# --- Run Class-wise Experiment
results = class_wise_active_learning_loop(
    project_name="flower102_class_wise_active_learning",
    model_class=SimpleCNN,
    dataset=flower_train,  # full training set used
    test_dataset=flower_test,  # test set filtered to selected classes
    strategies=strategies,
    query_size=QUERY_SIZE,
    max_iterations=MAX_ITERATIONS,
    device=DEVICE
)

# --- Plotting
plot_class_accuracy_vs_data_percentage(
    results,
    output_dir="thesis/results/flowers/all_strategies"
)
