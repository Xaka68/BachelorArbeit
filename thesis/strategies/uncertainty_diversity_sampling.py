import torch
import torch.nn as nn
import torch.nn.functional as F
from strategies.uncertainty_sampling import uncertainty_sampling
from strategies.constrastive_diversity_sampling import contrastive_diverse_sampling


def uncertainty_diversity_sampling(model, x_labeled, x_pool, query_size, device="cuda"):
    # Step 1: Uncertainty sampling to narrow down the pool
    top_k_entropy = min(5 * query_size, len(x_pool))
    entropy_top_indices = uncertainty_sampling(model, x_pool, top_k_entropy)

    # Step 2: Contrastive diversity on the high-entropy subset
    labeled_embeddings = None
    if x_labeled is not None and len(x_labeled) > 0:
        with torch.no_grad():
            labeled_embeddings = model(x_labeled.to(device)).detach().cpu()

    selected_indices = contrastive_diverse_sampling(
        model=model,
        x_pool=x_pool,
        selected_indices=entropy_top_indices,
        labeled_embeddings=labeled_embeddings,
        query_size=query_size,
        device=device
    )

    return selected_indices
