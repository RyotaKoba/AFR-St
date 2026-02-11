import numpy as np
import torch
from sklearn.mixture import BayesianGaussianMixture

def dpm_edge_outlier_removal(scores, alpha_edge=0.05, p_thr=0.8,
                              n_thr_ratio=0.01, n_thr_min=1,
                              K_max=5, weight_concentration_prior=0.001,fast_mode=True):
    """
    Dirichlet Process Mixture-based edge outlier removal for pruning scores
    
    Uses Bayesian GMM with Dirichlet Process prior to automatically determine
    the number of clusters. Small clusters at edges are treated as outliers.
    
    Key idea:
    - DP mixture automatically adapts number of clusters
    - Small clusters (few points) at edges are likely outliers
    - When no outliers exist, small clusters don't form
    - Multimodal distributions are naturally handled
    
    Parameters:
    -----------
    scores : torch.Tensor or np.ndarray, shape (4096, 14336)
        Pruning scores
    alpha_edge : float
        Edge region ratio (e.g., 0.05 = top/bottom 5% each)
    p_thr : float
        Probability threshold for outlier classification (e.g., 0.8-0.95)
    n_thr_ratio : float
        Cluster size threshold as ratio of total points (e.g., 0.01 = 1%)
    n_thr_min : int
        Minimum absolute cluster size threshold (e.g., 3 points)
    K_max : int
        Maximum number of mixture components (DP will use fewer if appropriate)
    weight_concentration_prior : float
        Controls cluster formation (smaller = fewer clusters, more sparse)
        Values like 0.001-0.01 work well for outlier detection
    
    Returns:
    --------
    cleaned_scores : same type as input
        Scores with outliers replaced by boundary values
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
    
    # Step 2: Fit Dirichlet Process Mixture (via Variational Bayesian GMM)
    # weight_concentration_prior_type='dirichlet_process' enables DP behavior
    # Very small weight_concentration_prior → fewer clusters, sparse solution
    dpm = BayesianGaussianMixture(
        n_components=K_max,
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=weight_concentration_prior,
        random_state=42,
        max_iter=100,
        n_init=5,
        verbose=0
    )
    
    X = x_sorted.reshape(-1, 1)
    dpm.fit(X)
    
    # Get converged parameters
    weights = dpm.weights_
    means = dpm.means_.flatten()
    covars = dpm.covariances_[:, 0, 0]
    
    # Filter to only active components (non-negligible weight)
    active_mask = weights > 1e-3
    n_active = np.sum(active_mask)
    
    print(f"\nDirichlet Process Mixture fitted:")
    print(f"  Active components: {n_active} / {K_max}")
    for k in range(K_max):
        if active_mask[k]:
            mu = means[k]
            sigma = np.sqrt(covars[k])
            pi = weights[k]
            print(f"  Component {k}: μ={mu:.6f}, σ={sigma:.6f}, π={pi:.4f}")
    
    # Step 3: Compute responsibilities and cluster assignments
    responsibilities = dpm.predict_proba(X)
    cluster_assignment = np.argmax(responsibilities, axis=1)
    
    # Step 4: Compute cluster sizes (expected number of points per cluster)
    cluster_sizes = np.array([
        np.sum(responsibilities[:, k]) for k in range(K_max)
    ])
    
    # Step 5: Identify small clusters as outlier candidates
    n_thr = max(n_thr_min, int(n_thr_ratio * N))
    small_clusters = np.where(cluster_sizes < n_thr)[0]
    
    print(f"\nCluster size analysis:")
    print(f"  Threshold: {n_thr} points ({n_thr_ratio*100:.1f}% or min {n_thr_min})")
    for k in range(K_max):
        if active_mask[k]:
            size = cluster_sizes[k]
            is_small = k in small_clusters
            marker = " [SMALL/OUTLIER]" if is_small else ""
            print(f"  Cluster {k}: {size:.1f} points{marker}")
    
    if len(small_clusters) == 0:
        print("\nNo small clusters detected - no outliers to remove")
        if is_torch:
            return scores
        else:
            return scores_np
    
    # Step 6: Compute outlier probability for each point
    # s_i = sum of responsibilities to small clusters
    outlier_prob = responsibilities[:, small_clusters].sum(axis=1)
    
    print(f"\nOutlier probability range: [{outlier_prob.min():.4f}, {outlier_prob.max():.4f}]")
    print(f"Points with p > {p_thr}: {np.sum(outlier_prob > p_thr):,}")
    
    # Step 7: Define edge masks
    i_edgeL = int(alpha_edge * N)
    i_edgeR = int((1 - alpha_edge) * N)
    
    edge_mask = np.zeros(N, dtype=bool)
    edge_mask[:i_edgeL] = True  # Left edge
    edge_mask[i_edgeR:] = True  # Right edge
    
    print(f"\nEdge regions (α={alpha_edge}):")
    print(f"  Left edge: indices [0, {i_edgeL}), values [{x_sorted[0]:.6f}, {x_sorted[i_edgeL-1]:.6f}]")
    print(f"  Right edge: indices [{i_edgeR}, {N}), values [{x_sorted[i_edgeR]:.6f}, {x_sorted[-1]:.6f}]")
    
    # Step 8: Find edge outliers (continuous from edges)
    is_outlier_candidate = outlier_prob > p_thr
    outlier_flag = np.zeros(N, dtype=bool)
    
    # Left edge outliers (continuous from left)
    j_L = -1
    for i in range(i_edgeL):
        if edge_mask[i] and is_outlier_candidate[i]:
            j_L = i
        else:
            break
    
    if j_L >= 0:
        outlier_flag[:j_L+1] = True
        print(f"\nLeft edge outliers: {j_L+1} points removed")
        print(f"  Range: [{x_sorted[0]:.6f}, {x_sorted[j_L]:.6f}]")
    else:
        print(f"\nLeft edge outliers: None")
    
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
    else:
        print(f"Right edge outliers: None")
    
    print(f"\nTotal outliers removed: {np.sum(outlier_flag):,} / {N:,} ({100*np.sum(outlier_flag)/N:.3f}%)")
    
    # Step 9: Map back to original indices
    outlier_flag_original = np.zeros(N, dtype=bool)
    outlier_flag_original[sort_indices] = outlier_flag
    
    # Step 10: Create cleaned scores
    cleaned_scores_np = scores_np.copy()
    mask_2d = ~outlier_flag_original.reshape(original_shape)
    
    if j_L >= 0:
        boundary_left = x_sorted[j_L]
        cleaned_scores_np[~mask_2d & (scores_np < boundary_left)] = boundary_left
    
    if j_R < N:
        boundary_right = x_sorted[j_R]
        cleaned_scores_np[~mask_2d & (scores_np > boundary_right)] = boundary_right
    
    # Convert back to torch if needed
    if is_torch:
        cleaned_scores = torch.from_numpy(cleaned_scores_np).to(original_device).to(original_dtype)
    else:
        cleaned_scores = cleaned_scores_np
    
    return cleaned_scores
