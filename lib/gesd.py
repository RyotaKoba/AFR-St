import torch
from tqdm import tqdm

def gesd_outlier_cleaning_torch(W_metric, alpha=0.05, max_outliers_ratio=0.05):
    """
    GESD Testで外れ値を検出し、マスクを返す (PyTorch版)
    
    Args:
        W_metric: torch.Tensor, shape (4096, 14336)
        alpha: float, 有意水準
        max_outliers_ratio: float, 最大外れ値比率
    
    Returns:
        cleaned_W_metric: torch.Tensor, shape (4096, 14336)
                         外れ値はnan
        mask: torch.Tensor, shape (4096, 14336), bool
              True=有効な値, False=外れ値
    """
    device = W_metric.device
    dtype = W_metric.dtype
    
    original_shape = W_metric.shape
    data_flat = W_metric.flatten()
    n = data_flat.numel()
    max_outliers = int(n * max_outliers_ratio)
    
    # 外れ値検出
    working_data = data_flat.clone()
    working_indices = torch.arange(n, device=device)
    outlier_indices = []
    
    for i in tqdm(range(max_outliers), desc="GESD Outlier Detection"):
        current_n = working_data.numel()
        
        mean = working_data.mean()
        std = working_data.std(unbiased=True)
        
        if std < 1e-10:
            print("Standard deviation too small, stopping GESD.")
            break
        
        # 最大偏差
        deviations = torch.abs(working_data - mean)
        max_idx = torch.argmax(deviations)
        max_dev = deviations[max_idx]
        
        # テスト統計量
        test_stat = max_dev / std
        
        # 臨界値計算 (t分布のパーセント点)
        # PyTorchにはt分布のppfがないので近似
        p = 1 - alpha / (2 * current_n)
        df = current_n - 2
        
        if df <= 0:
            print("Degrees of freedom <= 0, stopping GESD.")
            break
        
        # t分布の近似 (Wilson-Hilferty変換)
        # より正確にはscipyが必要だが、torchのみで近似
        # t_crit = t_distribution_ppf_approx(p, df)
        # lambda_crit = ((current_n - 1) * t_crit) / torch.sqrt(current_n * (current_n - 2 + t_crit**2))
        from scipy.stats import t
        t_crit = t.ppf(p, df)
        lambda_crit = ((current_n - 1) * t_crit) / torch.sqrt(torch.tensor(current_n * (current_n - 2 + t_crit**2))).item()
        
        # 仮説検定
        if test_stat > lambda_crit:
            original_idx = working_indices[max_idx]
            outlier_indices.append(original_idx.item())
            
            # 作業データから削除
            keep_mask = torch.ones(current_n, dtype=torch.bool, device=device)
            keep_mask[max_idx] = False
            working_data = working_data[keep_mask]
            working_indices = working_indices[keep_mask]
        else:
            print("No more outliers detected, stopping GESD.")
            break
    
    # マスク作成
    mask_flat = torch.ones(n, dtype=torch.bool, device=device)
    if len(outlier_indices) > 0:
        mask_flat[outlier_indices] = False
    
    mask = mask_flat.reshape(original_shape)
    
    # 外れ値をNaNで置換
    cleaned_data = W_metric.clone()
    cleaned_data[~mask] = float('nan')
    neuron_scores = torch.nanmean(cleaned_data, dim=0)
    return torch.abs(neuron_scores)


def t_distribution_ppf_approx(p, df):
    """
    t分布のパーセント点の近似 (PyTorch版)
    
    正規分布近似を使用
    dfが大きい場合は精度が高い
    """
    # 標準正規分布の近似
    z = torch.erfinv(torch.tensor(2 * p - 1)) * torch.sqrt(torch.tensor(2.0))
    
    # Cornish-Fisher展開による補正
    df_tensor = torch.tensor(float(df))
    g1 = (z**3 + z) / (4 * df_tensor)
    g2 = (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df_tensor**2)
    g3 = (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * df_tensor**3)
    
    t_approx = z + g1 + g2 + g3
    
    return t_approx
