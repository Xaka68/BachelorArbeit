import torch
import torch.nn as nn
import torch.nn.functional as F


def contrastive_diverse_sampling(
    model, x_pool, selected_indices, labeled_embeddings,
    query_size, projection_dim=128, device="cuda"
):
    model.eval()
    with torch.no_grad():
        embeddings = model(x_pool.to(device)).detach()

    # Define and apply projection head
    projection_head = nn.Sequential(
        nn.Linear(embeddings.size(1), projection_dim),
        nn.ReLU(),
        nn.Linear(projection_dim, projection_dim)
    ).to(device)

    projected = projection_head(embeddings)
    projected = F.normalize(projected, dim=1)

    selected_embeds = projected[selected_indices]

    if labeled_embeddings is not None and len(labeled_embeddings) > 0:
        with torch.no_grad():
            labeled_projected = projection_head(labeled_embeddings.to(device))
            labeled_projected = F.normalize(labeled_projected, dim=1)

        dists_to_labeled = torch.cdist(selected_embeds, labeled_projected, p=2)
        min_dists = dists_to_labeled.min(dim=1).values
        diversity_scores = min_dists
    else:
        diversity_scores = torch.ones(len(selected_indices), device=device)

    top_indices = torch.topk(diversity_scores, min(query_size, len(diversity_scores))).indices
    diverse_indices = [selected_indices[i.item()] for i in top_indices]

    return diverse_indices
