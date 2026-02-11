import numpy as np
import torch
from sklearn.mixture import BayesianGaussianMixture

def bmm_edge_outlier_removal(scores, K=3, alpha_edge=0.05, p_thr=0.8,
                              weight_concentration_prior=0.01,
                              check_bic=False, K_range=(2, 6),fast_mode=True):
    """
    Bayesian Mixture Model-based edge outlier removal for pruning scores
    
    Uses Variational Bayesian inference to fit a mixture model where
    low-weight components are treated as outlier/noise components.
    
    Parameters:
    -----------
    scores : torch.Tensor or np.ndarray, shape (4096, 14336)
        Pruning scores
    K : int
        Maximum number of mixture components
    alpha_edge : float
        Edge region ratio (e.g., 0.05 = top/bottom 5% each)
    p_thr : float
        Probability threshold for outlier classification (e.g., 0.8-0.95)
    weight_concentration_prior : float
        Dirichlet prior for mixture weights (small values favor sparse mixtures)
        Small values (e.g., 0.01) make outlier components naturally sparse
    check_bic : bool
        If True, select K using BIC
    K_range : tuple
        Range of K to test for BIC (min_K, max_K)
    
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
    
    # Step 2: Select K if requested
    if check_bic:
        K = select_K_by_bic_bayesian(x_sorted.reshape(-1, 1), K_range, weight_concentration_prior)
        print(f"Selected K by BIC: {K}")
    
    # Step 3: Fit Bayesian Gaussian Mixture
    # weight_concentration_prior: small values (e.g., 0.01) favor sparse solutions
    # - When outliers exist, they form a low-weight component
    # - When no outliers, that component naturally gets near-zero weight
    bgmm = BayesianGaussianMixture(
        n_components=K,
        weight_concentration_prior=weight_concentration_prior,
        random_state=42,
        max_iter=200,
        n_init=3
    )
    
    X = x_sorted.reshape(-1, 1)
    bgmm.fit(X)
    
    # Get converged weights (after VB inference)
    weights = bgmm.weights_
    means = bgmm.means_.flatten()
    covars = bgmm.covariances_[:, 0, 0]
    
    print(f"\nBayesian GMM fitted with K={K} components:")
    for k in range(K):
        mu = means[k]
        sigma = np.sqrt(covars[k])
        pi = weights[k]
        print(f"  Component {k}: μ={mu:.6f}, σ={sigma:.6f}, π={pi:.4f}")
    
    # Step 4: Compute responsibilities (posterior probabilities)
    responsibilities = bgmm.predict_proba(X)
    
    # Step 5: Identify noise/outlier components
    # Components with very low weight are likely noise components
    # Threshold: components with weight below certain percentile or absolute value
    weight_threshold = np.percentile(weights, 25)  # Bottom 25%
    noise_components = np.where(weights < max(weight_threshold, 0.05))[0]
    
    if len(noise_components) == 0:
        # No noise component identified - use lowest weight component
        noise_components = np.array([np.argmin(weights)])
    
    print(f"\nNoise components: {noise_components} (weights: {weights[noise_components]})")
    
    # Step 6: Compute outlier probability for each point
    # s_i = sum of responsibilities to noise components
    outlier_prob = responsibilities[:, noise_components].sum(axis=1)
    
    print(f"Outlier probability range: [{outlier_prob.min():.4f}, {outlier_prob.max():.4f}]")
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


def select_K_by_bic_bayesian(X, K_range=(2, 6), weight_concentration_prior=0.01):
    """Select optimal K using BIC for Bayesian GMM"""
    bic_scores = []
    K_values = list(range(K_range[0], K_range[1] + 1))
    
    print("\nBIC Selection (Bayesian GMM):")
    for K in K_values:
        bgmm = BayesianGaussianMixture(
            n_components=K,
            weight_concentration_prior=weight_concentration_prior,
            random_state=42,
            max_iter=200,
            n_init=3
        )
        bgmm.fit(X)
        bic = bgmm.bic(X)
        bic_scores.append(bic)
        print(f"  K={K}: BIC={bic:.2f}")
    
    best_K = K_values[np.argmin(bic_scores)]
    return best_K
