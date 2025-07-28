"""Common metric functions used by the 3GPP complexity scripts."""
from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import util
from sklearn.cluster import KMeans
from scipy.stats import entropy

__all__ = [
    "semantic_spread",
    "redundancy_index", 
    "cluster_entropy",
    "change_mag",
    "novelty_density",
    "get_optimal_cluster_count",
]

def _ensure_tensor(X: np.ndarray | torch.Tensor, device: str = 'cpu', dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert input to torch.Tensor with specified device and dtype."""
    if isinstance(X, np.ndarray):
        # Use compatible approach for older PyTorch versions
        t = torch.as_tensor(X, dtype=dtype)
        return t.to(device)
    return X.to(device=device, dtype=dtype)

def get_optimal_cluster_count(n_samples: int, min_clusters: int = 2, max_clusters: int = 50) -> int:
    """Calculate optimal number of clusters for a given sample size.
    
    Uses adaptive strategy based on dataset size with reasonable bounds.
    
    Args:
        n_samples: Number of samples in the dataset
        min_clusters: Minimum number of clusters (default: 2)
        max_clusters: Maximum number of clusters (default: 50)
        
    Returns:
        Optimal number of clusters
    """
    if n_samples < 10:
        return min(min_clusters, n_samples)
    elif n_samples < 50:
        return min(5, n_samples // 2)
    elif n_samples < 500:
        return min(15, n_samples // 10)
    else:
        # For larger datasets, use elbow rule with bounds
        k_sqrt = int(np.sqrt(n_samples / 2))
        return max(min_clusters, min(max_clusters, k_sqrt))

def semantic_spread(X: np.ndarray | torch.Tensor, device: str = 'cpu') -> float:
    """Return the total variance of the embedding matrix.
    
    More efficient implementation using direct variance calculation
    instead of building full covariance matrix.
    
    Args:
        X: Embedding matrix of shape (n_samples, n_features)
        device: Device for computation ('cpu' or 'cuda')
    """
    X_tensor = _ensure_tensor(X, device=device)
    # Direct variance calculation: sum of per-dimension variances
    return float(X_tensor.var(0, unbiased=False).sum())

def redundancy_index(X: np.ndarray | torch.Tensor, k: int = 5000, device: str = 'cpu',
                     rng: np.random.Generator | None = None) -> float:
    """Measure redundancy via average pairwise cosine similarity.

    Args:
        X: Embedding matrix of shape (n_samples, n_features)
        k: Maximum sample size for efficiency (default: 5000)
        device: Device for computation ('cpu' or 'cuda')
        rng: Random number generator for reproducible sampling
        
    Returns:
        Redundancy index: 1 - mean(cosine_similarity)
    """
    if rng is None:
        rng = np.random.default_rng()
        
    X_tensor = _ensure_tensor(X, device=device)
    
    if len(X_tensor) > k:
        indices = rng.choice(len(X_tensor), k, replace=False)
        X_tensor = X_tensor[indices]
    
    # Compute cosine similarity matrix
    sims = util.cos_sim(X_tensor, X_tensor)
    
    # Use memory-efficient upper triangular indexing
    n = sims.size(0)
    idx = torch.triu_indices(n, n, 1, device=sims.device)
    return 1.0 - sims[idx[0], idx[1]].mean().item()

def cluster_entropy(X: np.ndarray | torch.Tensor, sample: int = 5000, device: str = 'cpu',
                    rng: np.random.Generator | None = None) -> float:
    """Entropy of KMeans clusters drawn from X.
    
    Uses adaptive cluster count based on dataset size with proper bounds checking.
    
    Args:
        X: Embedding matrix of shape (n_samples, n_features)
        sample: Maximum sample size for efficiency (default: 5000)
        device: Device for computation ('cpu' or 'cuda')
        rng: Random number generator for reproducible sampling
        
    Returns:
        Cluster entropy in bits (base 2)
    """
    if rng is None:
        rng = np.random.default_rng()
        
    X_tensor = _ensure_tensor(X, device=device)
    
    if len(X_tensor) > sample:
        indices = rng.choice(len(X_tensor), sample, replace=False)
        X_tensor = X_tensor[indices]
    
    n = len(X_tensor)
    if n < 2:
        return 0.0  # Cannot cluster less than 2 samples
    
    # Get optimal cluster count with proper bounds
    n_clusters = min(get_optimal_cluster_count(n), n)
    if n_clusters < 2:
        return 0.0
    
    try:
        # Convert to numpy for scikit-learn compatibility
        X_np = X_tensor.cpu().numpy()
        
        # Use explicit n_init for version compatibility
        # Remove algorithm parameter for scikit-learn >=1.5 compatibility
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=10,  # Explicit value for older scikit-learn versions
            max_iter=300,
            random_state=0
        )
        labels = kmeans.fit_predict(X_np)
        
        # Calculate cluster distribution
        p = np.bincount(labels, minlength=n_clusters) / n
        
        # Remove zero probabilities to avoid log(0)
        p = p[p > 0]
        if len(p) <= 1:
            return 0.0
            
        return float(entropy(p, base=2))
        
    except Exception as e:
        # Log the error but don't fail silently
        import warnings
        warnings.warn(f"Cluster entropy calculation failed: {e}")
        return float('nan')


def change_mag(A: np.ndarray | torch.Tensor, B: np.ndarray | torch.Tensor, 
               device: str = 'cpu') -> float:
    """Cosine distance between mean embeddings of two sets.
    
    Args:
        A: First set of embeddings, shape (n_samples_A, n_features) or (n_features,) for centroid
        B: Second set of embeddings, shape (n_samples_B, n_features) or (n_features,) for centroid
        device: Device for computation ('cpu' or 'cuda')
        
    Returns:
        Change magnitude: 1 - cosine_similarity(mean(A), mean(B))
    """
    A_tensor = _ensure_tensor(A, device=device)
    B_tensor = _ensure_tensor(B, device=device)
    
    # Handle both sentence collections and pre-computed centroids
    if A_tensor.ndim == 1:
        mu_A = A_tensor.unsqueeze(0)  # Convert 1D centroid to (1, n_features)
    else:
        mu_A = A_tensor.mean(0, keepdim=True)  # Compute centroid from collection
    
    if B_tensor.ndim == 1:
        mu_B = B_tensor.unsqueeze(0)  # Convert 1D centroid to (1, n_features)
    else:
        mu_B = B_tensor.mean(0, keepdim=True)  # Compute centroid from collection
    
    # Cosine distance between centroids
    return 1.0 - util.cos_sim(mu_A, mu_B).item()

def novelty_density(Xp: np.ndarray | torch.Tensor, Xn: np.ndarray | torch.Tensor, 
                    k: int = 5000, batch_size: int = 512, device: str = 'cpu',
                    rng: np.random.Generator | None = None, 
                    metric: str = 'euclidean') -> float:
    """Proportion of novel content in Xn relative to reference pool Xp.
    
    Args:
        Xp: Previous/reference embeddings pool, shape (n_ref, n_features)
        Xn: New embeddings to check for novelty, shape (n_new, n_features)
        k: Maximum sample size for efficiency (default: 5000)
        batch_size: Batch size for memory-efficient computation (default: 512)
        device: Device for computation ('cpu' or 'cuda')
        rng: Random number generator for reproducible sampling
        metric: Distance metric ('euclidean' or 'cosine')
        
    Returns:
        Average minimum distance from new samples to reference pool
    """
    if rng is None:
        rng = np.random.default_rng()
    
    Xp_tensor = _ensure_tensor(Xp, device=device)
    Xn_tensor = _ensure_tensor(Xn, device=device)
    
    # Handle empty sets
    if len(Xn_tensor) == 0 or len(Xp_tensor) == 0:
        return float('nan')
    
    # Sample new embeddings if too large
    if len(Xn_tensor) > k:
        indices = rng.choice(len(Xn_tensor), k, replace=False)
        Xn_tensor = Xn_tensor[indices]
    
    # Sample reference pool (allow larger reference for better coverage)
    max_ref_size = k * 3
    if len(Xp_tensor) > max_ref_size:
        indices = rng.choice(len(Xp_tensor), max_ref_size, replace=False)
        Xp_tensor = Xp_tensor[indices]
    
    # Compute minimum distances in batches to save memory
    min_distances = []
    
    # GPU memory protection: fallback to CPU for large datasets
    if device == 'cuda' and len(Xn_tensor) * len(Xp_tensor) > 10_000_000:
        import warnings
        warnings.warn(f"Large distance matrix ({len(Xn_tensor)}×{len(Xp_tensor)}), switching to CPU to avoid OOM")
        Xn_tensor = Xn_tensor.cpu()
        Xp_tensor = Xp_tensor.cpu()
        device = 'cpu'
    
    for start in range(0, len(Xn_tensor), batch_size):
        end = min(start + batch_size, len(Xn_tensor))
        batch_Xn = Xn_tensor[start:end]
        
        if metric == 'euclidean':
            # Use torch.cdist for efficient pairwise Euclidean distance
            distances = torch.cdist(batch_Xn, Xp_tensor, p=2)
        elif metric == 'cosine':
            # Convert to cosine distance: 1 - cosine_similarity
            similarities = util.cos_sim(batch_Xn, Xp_tensor)
            distances = 1.0 - similarities
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        # Find minimum distance for each sample in the batch
        batch_min_distances = distances.min(dim=1).values
        min_distances.append(batch_min_distances)
        
        # Clean up GPU memory for large batches
        if device == 'cuda':
            del distances
            if 'similarities' in locals():
                del similarities
            torch.cuda.empty_cache()
    
    # Concatenate all minimum distances and compute mean
    all_min_distances = torch.cat(min_distances)
    return float(all_min_distances.mean())

