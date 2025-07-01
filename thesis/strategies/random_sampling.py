import numpy as np

def random_sampling(model, x_pool, query_size):
    indices = np.random.choice(len(x_pool), size=min(query_size, len(x_pool)), replace=False)
    return indices
