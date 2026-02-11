import numpy as np
import torch
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

def gmm_edge_outlier_removal(scores, K=3, alpha_edge=0.05, q_tail=0.005, 
                              use_density=True, z_thr=4.0, 
                              check_bic=False, K_range=(2, 6)):
    """
    GMM-based edge outlier removal for pruning scores
    
    Parameters:
    -----------
    scores : torch.Tensor or np.ndarray, shape (4096, 14336)
        Pruning scores (e.g., FO scores, SNIP scores)
        Can be on CUDA device - will be handled automatically
    K : int
        Number of GMM components (clusters)
    alpha_edge : float
        Edge region ratio (e.g., 0.05 = top/bottom 5% each)
    q_tail : float
        Quantile threshold for low density (e.g., 0.005 = bottom 0.5%)
    use_density : bool
        If True, use density-based outlier score
        If False, use distance-based (z-score) outlier score
    z_thr : float
        Z-score threshold for distance-based method
    check_bic : bool
        If True, select K using BIC
    K_range : tuple
        Range of K to test for BIC (min_K, max_K)
    
    Returns:
    --------
    cleaned_scores : same type as input (torch.Tensor or np.ndarray)
        Scores with outliers replaced by the boundary values
    mask : same type as input, dtype=bool
        True for inliers, False for outliers
    stats : dict
        Statistics and diagnostics
    """
    
    # Handle torch.Tensor input
    is_torch = isinstance(scores, torch.Tensor)
    original_device = None
    
    if is_torch:
        original_device = scores.device
        original_dtype = scores.dtype
        scores_np = scores.detach().cpu().numpy()
    else:
        scores_np = scores
    
    original_shape = scores_np.shape
    
    # Step 1: Flatten and sort
    x = scores_np.flatten()
    N = len(x)
    x_sorted = np.sort(x)
    sort_indices = np.argsort(x)
    
    print(f"Total data points: {N:,}")
    print(f"Score range: [{x.min():.6f}, {x.max():.6f}]")
    
    # Step 2: Fit GMM (with optional BIC selection)
    if check_bic:
        K = select_K_by_bic(x_sorted.reshape(-1, 1), K_range)
        print(f"Selected K by BIC: {K}")    
    
    gmm = GaussianMixture(n_components=K, random_state=42, max_iter=200)
    gmm.fit(x_sorted.reshape(-1, 1))
    
    print(f"\nGMM fitted with K={K} components:")
    for k in range(K):
        mu = gmm.means_[k, 0]
        sigma = np.sqrt(gmm.covariances_[k, 0, 0])
        pi = gmm.weights_[k]
        print(f"  Component {k}: μ={mu:.6f}, σ={sigma:.6f}, π={pi:.4f}")
    
    # Step 3: Compute density and cluster assignments
    log_density = gmm.score_samples(x_sorted.reshape(-1, 1))
    density = np.exp(log_density)
    
    responsibilities = gmm.predict_proba(x_sorted.reshape(-1, 1))
    cluster_assignment = np.argmax(responsibilities, axis=1)
    
    # Step 4: Compute outlier scores
    if use_density:
        # Pattern A: Density-based
        outlier_scores = -log_density  # Higher = more outlier-ish
        tau = np.quantile(density, q_tail)
        is_outlier_candidate = density < tau
        print(f"\nDensity-based threshold (q={q_tail}): {tau:.6e}")
    else:
        # Pattern B: Distance-based (z-score from assigned cluster)
        z_scores = np.zeros(N)
        for i in range(N):
            k = cluster_assignment[i]
            mu_k = gmm.means_[k, 0]
            sigma_k = np.sqrt(gmm.covariances_[k, 0, 0])
            z_scores[i] = np.abs(x_sorted[i] - mu_k) / sigma_k
        
        outlier_scores = z_scores
        is_outlier_candidate = z_scores > z_thr
        print(f"\nDistance-based threshold (z>{z_thr}): {np.sum(is_outlier_candidate)} candidates")
    
    # Step 5: Define edge masks
    i_edgeL = int(alpha_edge * N)
    i_edgeR = int((1 - alpha_edge) * N)
    
    edge_mask = np.zeros(N, dtype=bool)
    edge_mask[:i_edgeL] = True  # Left edge
    edge_mask[i_edgeR:] = True  # Right edge
    
    print(f"\nEdge regions (α={alpha_edge}):")
    print(f"  Left edge: indices [0, {i_edgeL}), values [{x_sorted[0]:.6f}, {x_sorted[i_edgeL-1]:.6f}]")
    print(f"  Right edge: indices [{i_edgeR}, {N}), values [{x_sorted[i_edgeR]:.6f}, {x_sorted[-1]:.6f}]")
    
    # Step 6: Find edge outliers (continuous from edges)
    outlier_flag = np.zeros(N, dtype=bool)
    
    # Left edge outliers (continuous from left)
    j_L = 0
    for i in range(i_edgeL):
        if edge_mask[i] and is_outlier_candidate[i]:
            j_L = i
        else:
            break
    
    if j_L > 0:
        outlier_flag[:j_L+1] = True
        print(f"\nLeft edge outliers: {j_L+1} points removed")
        print(f"  Range: [{x_sorted[0]:.6f}, {x_sorted[j_L]:.6f}]")
    
    # Right edge outliers (continuous from right)
    j_R = N
    for i in range(N-1, i_edgeR-1, -1):
        if edge_mask[i] and is_outlier_candidate[i]:
            j_R = i
        else:
            break
    
    if j_R < N:
        outlier_flag[j_R:] = True
        print(f"Right edge outliers: {N - j_R} points removed")
        print(f"  Range: [{x_sorted[j_R]:.6f}, {x_sorted[-1]:.6f}]")
    
    print(f"\nTotal outliers removed: {np.sum(outlier_flag):,} / {N:,} ({100*np.sum(outlier_flag)/N:.3f}%)")
    
    # Step 7: Map back to original indices
    outlier_flag_original = np.zeros(N, dtype=bool)
    outlier_flag_original[sort_indices] = outlier_flag
    
    # Step 8: Create cleaned scores (replace outliers with boundary values)
    cleaned_scores_np = scores_np.copy()
    mask_2d = ~outlier_flag_original.reshape(original_shape)
    
    if j_L > 0:
        boundary_left = x_sorted[j_L]
        cleaned_scores_np[~mask_2d & (scores_np < boundary_left)] = boundary_left
    
    if j_R < N:
        boundary_right = x_sorted[j_R]
        cleaned_scores_np[~mask_2d & (scores_np > boundary_right)] = boundary_right
    
    # Statistics
    stats = {
        'K': K,
        'n_outliers': np.sum(outlier_flag),
        'outlier_ratio': np.sum(outlier_flag) / N,
        'left_boundary': x_sorted[j_L] if j_L > 0 else None,
        'right_boundary': x_sorted[j_R] if j_R < N else None,
        'gmm_means': gmm.means_.flatten(),
        'gmm_stds': np.sqrt(gmm.covariances_[:, 0, 0]),
        'gmm_weights': gmm.weights_,
    }
    
    # Convert back to torch if input was torch
    if is_torch:
        cleaned_scores = torch.from_numpy(cleaned_scores_np).to(original_device).to(original_dtype)
        mask = torch.from_numpy(mask_2d).to(original_device)
    else:
        cleaned_scores = cleaned_scores_np
        mask = mask_2d
    
    return cleaned_scores, mask, stats


def select_K_by_bic(X, K_range=(2, 6)):
    """Select optimal K using BIC"""
    bic_scores = []
    K_values = list(range(K_range[0], K_range[1] + 1))
    
    print("\nBIC Selection:")
    for K in K_values:
        gmm = GaussianMixture(n_components=K, random_state=42, max_iter=200)
        gmm.fit(X)
        bic = gmm.bic(X)
        bic_scores.append(bic)
        print(f"  K={K}: BIC={bic:.2f}")
    
    best_K = K_values[np.argmin(bic_scores)]
    return best_K
