import torch
from .gmm import gmm_edge_outlier_removal, select_K_by_bic
from .gesd import gesd_outlier_cleaning_torch
from .kde import kde_edge_outlier_removal
from .dpm import dpm_edge_outlier_removal
from .bmm import bmm_edge_outlier_removal

def calculate_neuron_score(method ,W_metric):
    if method is None:
        return W_metric
    elif method == "percentile":
        # Method 1: Trim x%
        trim_percent = 2
        sorted_W, _ = torch.sort(W_metric, dim=0)
        n_rows = W_metric.shape[0]
        trim_count = int(n_rows * trim_percent / 100)
        cleaned_scores = sorted_W[trim_count:-trim_count, :]
    elif method == "gmm":
        # Method 2: GMM Trim
        cleaned_scores, _, _= gmm_edge_outlier_removal(W_metric, K=3, alpha_edge=0.05,q_tail=0.02, use_density=True, check_bic=True, K_range=(1, 5))
    elif method == "kde":
        # Method 3: KDE Trim
        cleaned_scores = kde_edge_outlier_removal(W_metric)
    elif method == "gesd":
        # Method 4: GESD Trim
        cleaned_scores = gesd_outlier_cleaning_torch(W_metric)
    elif method == "dpm":
        # Method 5: DPM Trim
        cleaned_scores = dpm_edge_outlier_removal(W_metric)
    elif method == "bmm":
        # Method 6: BMM Trim
        cleaned_scores = bmm_edge_outlier_removal(W_metric)

    return cleaned_scores

def weight_wise_to_neuron_wise(scores):
    # ===========================================================
    # Method 1: MeanAbs
    # ==========================================================~
    mean_scores = scores.mean(axis=0)
    return torch.abs(mean_scores)
    
