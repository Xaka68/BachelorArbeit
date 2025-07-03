import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
import logging
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
from collections import Counter

# --- Load AG_NEWS dataset
dataset = load_dataset("fancyzhx/ag_news")
train_data = dataset["train"]
test_data = dataset["test"]

# --- Tokenization using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

MAX_LEN = 128

def tokenize(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

# --- Custom Dataset wrapper
class AGNewsDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer):
        self.tokenizer = tokenizer
        self.dataset = hf_dataset
        self.label_map = {label: i for i, label in enumerate(sorted(set(hf_dataset["label"])))}

    def __getitem__(self, idx):
        example = self.dataset[idx]
        encoding = self.tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return (
            encoding["input_ids"].squeeze(0),
            encoding["attention_mask"].squeeze(0),
            int(example["label"])
        )

    def __len__(self):
        return len(self.dataset)

# --- Dataset preparation
train_dataset = AGNewsDataset(train_data, tokenizer)
test_dataset = AGNewsDataset(test_data, tokenizer)

# --- Simple BERT Classifier
class BERTClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output
        return self.classifier(self.dropout(cls_output))

# --- Wrappers and fixes for your strategies to accept BERT input format
# Make sure each strategy takes batches of input_ids and attention_mask

# --- Active Learning Imports
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

# --- Define strategies
strategies = {
    "Random": random_sampling,
    "Uncertainty": uncertainty_sampling,
    "Bayesian": bayesian_sampling,
    "Uncertainty+Diversity": uncertainty_diversity_sampling,
    "Bayesian+Diversity": bayesian_diversity_sampling
}

# --- Run Experiment
results = class_wise_active_learning_loop(
    project_name="agn_class_wise_active_learning",
    model_class=BERTClassifier,
    dataset=train_dataset,
    test_dataset=test_dataset,
    strategies=strategies,
    query_size=QUERY_SIZE,
    device=DEVICE
)

# --- Plot results
plot_class_accuracy_vs_data_percentage(results, output_dir="thesis/results/AGN/all_strategies")
