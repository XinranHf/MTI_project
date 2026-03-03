"""
Plot results from the SPA algorithm.

This module provides visualization functions for the SPA algorithm outputs.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_RESULT(img_noise, img_original, X_MC, N_burn_in, N):
    """
    Plot results from the SPA algorithm.
    
    Parameters
    ----------
    ing_noise : ndarray
        Noisy observations.
    img_original : ndarray
        Original image.
    X_MC : ndarray
        MCMC samples of x (N x N x N_MC).
    N_burn_in : int
        Number of burn-in iterations.
    N : int
        Image dimension (N x N).
    """
    
    plt.close('all')
    
    # 1. PLOT ORIGINAL, OBSERVATIONS AND ESTIMATES
    
    img_mmse = np.mean(X_MC[:, :, N_burn_in:], axis=2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    images = [img_original, img_noise, img_mmse]
    titles = ['Original Image', 'Blurred & Noisy Observation', 'MMSE Estimate of x']

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis('equal')
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    
    # # Plot the MMSE of z
    # plt.figure(4, figsize=(8, 8))
    # plt.imshow(np.mean(Z_MC[:, :, N_burn_in:], axis=2), cmap='gray', vmin=0, vmax=255)
    # plt.axis('equal')
    # plt.axis('off')
    # plt.title('MMSE estimate of z')
    # plt.tight_layout()
    
    # # Plot the MMSE of u
    # plt.figure(5, figsize=(8, 8))
    # im = plt.imshow(np.mean(U_MC[:, :, N_burn_in:], axis=2), cmap='gray')
    # plt.colorbar(im)
    # plt.axis('equal')
    # plt.axis('off')
    # plt.title('MMSE estimate of u')
    # plt.tight_layout()
    
    # 2. PLOT THE 90% CREDIBILITY INTERVALS
    CI_90 = np.zeros((N, N))
    samples = X_MC[:, :, N_burn_in:]  # shape (N, N, N_samples)
    quant_5  = np.percentile(samples, 5,  axis=2)
    quant_95 = np.percentile(samples, 95, axis=2)
    CI_90 = np.abs(quant_95 - quant_5)
    
    plt.figure(6, figsize=(8, 8))
    im = plt.imshow(CI_90, cmap='gray_r')
    plt.colorbar(im)
    plt.axis('equal')
    plt.axis('off')
    plt.title('90% credibility intervals')
    plt.tight_layout()
    
    plt.show()
