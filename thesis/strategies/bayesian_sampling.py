import torch
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def bayesian_sampling(model, x_pool, n, dropout_passes=10, batch_size=32):
    model.train()
    num_samples = len(x_pool)
    num_classes = 10
    probs_sum = torch.zeros((num_samples, num_classes)).to(device)

    for _ in range(dropout_passes):
        current_probs = []
        for i in range(0, num_samples, batch_size):
            batch = x_pool[i:i+batch_size].to(device)
            with torch.no_grad():
                out = model(batch)
                prob = F.softmax(out, dim=1)
            current_probs.append(prob)
        current_probs = torch.cat(current_probs, dim=0)
        probs_sum += current_probs

    avg_probs = probs_sum / dropout_passes
    entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=1)
    return torch.topk(entropy, k=n).indices.cpu().numpy()
