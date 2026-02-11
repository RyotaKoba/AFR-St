import numpy as np
import torch
from scipy.fft import fft, ifft


def fft_kde_1d(x, bandwidth, grid_size=2**16):
    """
    Fast 1D KDE using FFT convolution
    
    O(N log N) complexity instead of O(N^2)
    
    Parameters:
    -----------
    x : np.ndarray, shape (N,)
        Data points (already sorted is best)
    bandwidth : float
        Gaussian kernel bandwidth
    grid_size : int
        Number of grid points (power of 2 for FFT efficiency)
    
    Returns:
    --------
    x_grid : np.ndarray
        Grid points where density is evaluated
    density_grid : np.ndarray
        Density values at grid points
    """
    N = len(x)
    
    # Define grid over data range with some padding
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    padding = 3 * bandwidth  # 3-sigma padding
    
    # Create grid with grid_size points
    x_grid = np.linspace(x_min - padding, x_max + padding, grid_size)
    dx = x_grid[1] - x_grid[0]
    
    # Bin the data onto the grid (histogram)
    # Important: bins=grid_size gives grid_size-1 bins, but we want grid_size
    hist, bin_edges = np.histogram(x, bins=grid_size, range=(x_min - padding, x_max + padding))
    
    # Normalize histogram to get empirical density
    hist = hist.astype(np.float64) / (N * dx)
    
    # Pad hist to grid_size if needed (histogram returns grid_size-1 values)
    if len(hist) < grid_size:
        hist = np.pad(hist, (0, grid_size - len(hist)), mode='constant', constant_values=0)
    
    # Create Gaussian kernel
    kernel_x = np.arange(grid_size)
    kernel_x = kernel_x - grid_size // 2
    kernel_x = kernel_x * dx
    kernel = (1 / (bandwidth * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (kernel_x / bandwidth)**2)
    
    # Ensure kernel is normalized
    kernel = kernel / (kernel.sum() * dx)
    
    # FFT convolution: density = histogram ⊗ kernel
    hist_fft = fft(hist)
    kernel_fft = fft(np.fft.ifftshift(kernel))
    
    density_fft = hist_fft * kernel_fft
    density_grid = np.real(ifft(density_fft))
    
    # Output grid matches the histogram bins
    x_grid_out = bin_edges[:-1] + dx / 2  # Center of each bin
    
    # Ensure same length
    if len(density_grid) > len(x_grid_out):
        density_grid = density_grid[:len(x_grid_out)]
    
    return x_grid_out, density_grid


def interpolate_density(x_data, x_grid, density_grid):
    """
    Interpolate density values at data points
    
    Parameters:
    -----------
    x_data : np.ndarray
        Points where we want density values
    x_grid : np.ndarray
        Grid points where density was evaluated
    density_grid : np.ndarray
        Density values at grid points
    
    Returns:
    --------
    density_at_data : np.ndarray
        Interpolated density at x_data points
    """
    # Linear interpolation
    density_at_data = np.interp(x_data, x_grid, density_grid)
    return density_at_data


def kde_edge_outlier_removal(scores, alpha_center=0.1, alpha_edge=0.05,
                              q_density=0.01, bandwidth='scott',
                              use_quantile_threshold=True, c_density=0.1,
                              grid_size=2**16):
    """
    KDE-based edge outlier removal for pruning scores
    
    Uses FFT-based KDE for O(N log N) complexity instead of O(N^2).
    Efficient for large datasets (millions of points).
    
    Parameters:
    -----------
    scores : torch.Tensor or np.ndarray, shape (4096, 14336)
        Pruning scores
    alpha_center : float
        Central region ratio for typical density estimation (e.g., 0.1-0.2)
    alpha_edge : float
        Edge region ratio for scanning (e.g., 0.05-0.1)
    q_density : float
        Quantile threshold for low density (e.g., 0.01-0.05)
    bandwidth : str or float
        KDE bandwidth ('scott', 'silverman', or float value)
    use_quantile_threshold : bool
        If True, use Pattern B (quantile-based threshold)
        If False, use Pattern A (ratio-based threshold)
    c_density : float
        Ratio for Pattern A threshold (e.g., 0.1-0.2)
    grid_size : int
        Number of grid points for FFT (default: 2^16 = 65536)
        Should be power of 2 for FFT efficiency
        Larger = more accurate but slower
    
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
    
    # Step 2: KDE bandwidth estimation
    if bandwidth == 'scott':
        # Scott's rule of thumb
        std = np.std(x_sorted)
        h = 1.06 * std * N ** (-1/5)
    elif bandwidth == 'silverman':
        # Silverman's rule of thumb
        std = np.std(x_sorted)
        h = 0.9 * std * N ** (-1/5)
    else:
        h = float(bandwidth)
    
    print(f"\nKDE bandwidth (h): {h:.6e}")
    
    # Step 3: Compute KDE using FFT (fast!)
    print(f"Computing FFT-based KDE with grid_size={grid_size}...")
    x_grid, density_grid = fft_kde_1d(x_sorted, bandwidth=h, grid_size=grid_size)
    
    # Step 4: Interpolate density at each data point
    f = interpolate_density(x_sorted, x_grid, density_grid)
    
    print(f"Density range: [{f.min():.6e}, {f.max():.6e}]")
    
    # Step 5: Compute typical density from central region
    i_min = max(0, int(alpha_center * N))
    i_max = min(N-1, int((1 - alpha_center) * N))
    
    f_center = f[i_min:i_max+1]
    f_typ = np.median(f_center)
    
    print(f"\nCentral region: indices [{i_min}, {i_max}]")
    print(f"Typical density (median): {f_typ:.6e}")
    
    # Step 6: Determine threshold
    if use_quantile_threshold:
        # Pattern B: Quantile-based
        tau = np.quantile(f, q_density)
        print(f"Threshold (quantile q={q_density}): {tau:.6e}")
    else:
        # Pattern A: Ratio-based
        tau = c_density * f_typ
        print(f"Threshold (ratio c={c_density}): {tau:.6e}")
    
    is_low_density = f < tau
    
    # Step 7: Define edge masks
    i_edgeL = max(1, int(alpha_edge * N))
    i_edgeR = max(0, int((1 - alpha_edge) * N))
    
    print(f"\nEdge regions (α={alpha_edge}):")
    print(f"  Left edge: indices [0, {i_edgeL}), values [{x_sorted[0]:.6f}, {x_sorted[i_edgeL-1]:.6f}]")
    print(f"  Right edge: indices [{i_edgeR}, {N}), values [{x_sorted[i_edgeR]:.6f}, {x_sorted[-1]:.6f}]")
    
    # Step 8: Find edge outliers (continuous from edges)
    outlier_flag = np.zeros(N, dtype=bool)
    
    # Left edge outliers
    j_L = -1
    for i in range(i_edgeL):
        if is_low_density[i]:
            j_L = i
        else:
            break
    
    if j_L >= 0:
        outlier_flag[:j_L+1] = True
        print(f"\nLeft edge outliers: {j_L+1} points removed")
        print(f"  Range: [{x_sorted[0]:.6f}, {x_sorted[j_L]:.6f}]")
    else:
        print(f"\nLeft edge outliers: None")
    
    # Right edge outliers
    j_R = N
    for i in range(N-1, i_edgeR-1, -1):
        if is_low_density[i]:
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