import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def uncertainty_sampling(model, x_pool, n, batch_size=32):
    model.eval()
    entropy_scores = []
    with torch.no_grad():
        for i in range(0, len(x_pool), batch_size):
            batch = x_pool[i:i+batch_size].to(device)
            outputs = model(batch)
            probs = F.softmax(outputs, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            entropy_scores.append(entropy)
    if len(entropy_scores) == 0:
        return np.array([])
    entropy_scores = torch.cat(entropy_scores)
    k = min(n, len(entropy_scores))
    return torch.topk(entropy_scores, k=k).indices.cpu().numpy()


def compute_entropy(model, x_pool, batch_size=32):
    model.eval()
    entropy_scores = []
    with torch.no_grad():
        for i in range(0, len(x_pool), batch_size):
            batch = x_pool[i:i+batch_size].to(device)
            outputs = model(batch)
            probs = F.softmax(outputs, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            entropy_scores.append(entropy)
    if len(entropy_scores) == 0:
        return torch.tensor([])
    return torch.cat(entropy_scores)
