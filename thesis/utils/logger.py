# utils/logger.py
import os
from datetime import datetime

def create_experiment_folder(base_path="results"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base_path, timestamp)
    os.makedirs(path, exist_ok=True)
    return path

def save_config(config, path):
    with open(os.path.join(path, 'config.yaml'), 'w') as f:
        f.write(str(config))
