import numpy as np
import ot


def cal_2_wasserstein_dist(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Calculate 2-Wasserstein distance between two point clouds X and Y.
    
    WD(X,Y) = (min_{p∈Π(px,py)} Σ pij ||xi - yj||²)^(1/2)
    
    Uses optimal transport (ot.emd2) to compute the distance.
    
    Args:
        X: numpy array of shape (n, d) - first point cloud
        Y: numpy array of shape (m, d) - second point cloud
    
    Returns:
        float: 2-Wasserstein distance
    """
    # Check for NaN or Inf values
    if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
        print("Warning: NaN detected in samples for Wasserstein distance")
        return float('nan')
    if np.any(np.isinf(X)) or np.any(np.isinf(Y)):
        print("Warning: Inf detected in samples for Wasserstein distance")
        return float('nan')
    
    n = X.shape[0]
    m = Y.shape[0]
    
    a = np.ones(n) / n
    b = np.ones(m) / m
    M = ot.dist(X, Y, metric='sqeuclidean')
    
    if np.any(np.isnan(M)) or np.any(np.isinf(M)):
        print("Warning: NaN/Inf in distance matrix")
        return float('nan')
    
    try:
        wasserstein_dist_squared = ot.emd2(a, b, M, numItermax=int(1e7))
    except Exception as e:
        print(f"Warning: Wasserstein distance computation failed: {e}")
        return float('nan')
    
    return np.sqrt(max(0, wasserstein_dist_squared))


def cal_energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Calculate Energy Distance between two point clouds X and Y.
    
    ED(X,Y) = (2/nm) Σ||xi - yj||2 - (1/n²) Σ||xi - xj||2 - (1/m²) Σ||yi - yj||2
    
    Args:
        X: numpy array of shape (n, d) - first point cloud
        Y: numpy array of shape (m, d) - second point cloud
    
    Returns:
        float: Energy distance
    """
    # Check for NaN or Inf values
    if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
        print("Warning: NaN detected in samples for Energy distance")
        return float('nan')
    if np.any(np.isinf(X)) or np.any(np.isinf(Y)):
        print("Warning: Inf detected in samples for Energy distance")
        return float('nan')
    
    n = X.shape[0]
    m = Y.shape[0]
    
    diff_XY = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    dist_XY = np.linalg.norm(diff_XY, axis=2)
    term1 = (2.0 / (n * m)) * np.sum(dist_XY)
    
    diff_XX = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_XX = np.linalg.norm(diff_XX, axis=2)
    term2 = (1.0 / (n * n)) * np.sum(dist_XX)
    
    diff_YY = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]
    dist_YY = np.linalg.norm(diff_YY, axis=2)
    term3 = (1.0 / (m * m)) * np.sum(dist_YY)
    
    energy_dist = term1 - term2 - term3
    
    return energy_dist
