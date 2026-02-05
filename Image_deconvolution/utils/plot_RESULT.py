"""
Plot results from the SPA algorithm.

This module provides visualization functions for the SPA algorithm outputs.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_RESULT(y, refl, X_MC, Z_MC, U_MC, N_bi, N_MC, N):
    """
    Plot results from the SPA algorithm.
    
    Parameters
    ----------
    y : ndarray
        Noisy observations.
    refl : ndarray
        Original image.
    X_MC : ndarray
        MCMC samples of x (N x N x N_MC).
    Z_MC : ndarray
        MCMC samples of z (N x N x N_MC).
    U_MC : ndarray
        MCMC samples of u (N x N x N_MC).
    N_bi : int
        Number of burn-in iterations.
    N_MC : int
        Total number of MCMC iterations.
    N : int
        Image dimension (N x N).
    """
    
    plt.close('all')
    
    # 1. PLOT ORIGINAL, OBSERVATIONS AND ESTIMATES
    
    # Plot the original image
    plt.figure(1, figsize=(8, 8))
    plt.imshow(refl, cmap='gray', vmin=0, vmax=255)
    plt.axis('equal')
    plt.axis('off')
    plt.title('Original image')
    plt.tight_layout()
    
    # Plot the noisy observation
    plt.figure(2, figsize=(8, 8))
    plt.imshow(y, cmap='gray', vmin=0, vmax=255)
    plt.axis('equal')
    plt.axis('off')
    plt.title('Blurred and noisy observation')
    plt.tight_layout()
    
    # Plot the MMSE of x
    plt.figure(3, figsize=(8, 8))
    plt.imshow(np.mean(X_MC[:, :, N_bi:], axis=2), cmap='gray', vmin=0, vmax=255)
    plt.axis('equal')
    plt.axis('off')
    plt.title('MMSE estimate of x')
    plt.tight_layout()
    
    # Plot the MMSE of z
    plt.figure(4, figsize=(8, 8))
    plt.imshow(np.mean(Z_MC[:, :, N_bi:], axis=2), cmap='gray', vmin=0, vmax=255)
    plt.axis('equal')
    plt.axis('off')
    plt.title('MMSE estimate of z')
    plt.tight_layout()
    
    # Plot the MMSE of u
    plt.figure(5, figsize=(8, 8))
    im = plt.imshow(np.mean(U_MC[:, :, N_bi:], axis=2), cmap='gray')
    plt.colorbar(im)
    plt.axis('equal')
    plt.axis('off')
    plt.title('MMSE estimate of u')
    plt.tight_layout()
    
    # 2. PLOT THE 90% CREDIBILITY INTERVALS
    CI_90 = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            arr = X_MC[i, j, N_bi:].reshape(-1)
            quant_5 = np.percentile(arr, 5)
            quant_95 = np.percentile(arr, 95)
            CI_90[i, j] = abs(quant_95 - quant_5)
    
    plt.figure(6, figsize=(8, 8))
    im = plt.imshow(CI_90, cmap='gray_r')
    plt.colorbar(im)
    plt.axis('equal')
    plt.axis('off')
    plt.title('90% credibility intervals')
    plt.tight_layout()
    
    plt.show()
