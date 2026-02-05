"""
Improved Signal-to-Noise Ratio (ISNR) calculation.

This function computes the ISNR and MSE metrics for image quality assessment.
"""

import numpy as np


def ISNR(X, Y, X_MC, N_bi):
    """
    Compute Improved Signal-to-Noise Ratio (ISNR) and Mean Squared Error (MSE).
    
    Parameters
    ----------
    X : ndarray
        Original image.
    Y : ndarray
        Noisy/degraded observation.
    X_MC : ndarray
        MCMC samples (N x N x N_MC).
    N_bi : int
        Number of burn-in iterations to exclude.
    
    Returns
    -------
    isnr : float
        Improved Signal-to-Noise Ratio in dB.
    mse : float
        Mean Squared Error.
    """
    
    # Compute MMSE estimate
    mmse_estimate = np.mean(X_MC[:, :, N_bi:], axis=2)
    
    # Compute MSE
    mse = np.linalg.norm(X - mmse_estimate, 'fro')**2 / X.size
    
    # Compute ISNR
    numerator = np.sum((Y.flatten() - X.flatten())**2)
    denominator = mse * X.size
    isnr = 10 * np.log10(numerator / denominator)
    
    return isnr, mse
