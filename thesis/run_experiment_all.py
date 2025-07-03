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
QUERY_SIZE = 500
MAX_ITERATIONS = 50
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

all_classes = sorted(set(label for _, label in mnist_train))

# --- Run Class-wise Experiment
results = class_wise_active_learning_loop(
    project_name="mnist_class_wise_active_learning",
    model_class=SimpleCNN,
    dataset=mnist_train,
    test_dataset=mnist_test,
    strategies=strategies,
    query_size=QUERY_SIZE,
    device=DEVICE
)

# --- Plotting
plot_class_accuracy_vs_data_percentage(
    results,
)