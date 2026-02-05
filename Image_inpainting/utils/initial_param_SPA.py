"""
Initial variables and parameters for SPA algorithm (Image Inpainting).

This module generates and saves initial parameters for the SPA algorithm
applied to image inpainting.
"""

import numpy as np
from scipy.sparse import eye as speye, spdiags
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from skimage import data, io
import warnings


def initialize_parameters():
    """
    Initialize all parameters for the SPA algorithm (image inpainting).
    
    Returns
    -------
    dict
        Dictionary containing all initialized parameters.
    """
    
    # Set random seed
    np.random.seed(1)
    
    # Load original image
    try:
        X = io.imread('cameraman.tif').astype(float)
        if len(X.shape) == 3:
            X = X[:, :, 0]  # Convert to grayscale if needed
    except:
        # Fallback to scikit-image's camera image
        print("Warning: cameraman.tif not found. Using scikit-image camera.")
        X = data.camera().astype(float)
    
    N = X.shape[0]
    
    # Random observations (image inpainting with 40% of the pixels missing)
    H = (np.random.rand(N, N) > 0.4).astype(float)  # 40% of the pixels missing
    Y = X * H
    
    # Set BSNR
    BSNR = 40  # SNR expressed in decibels
    P_signal = np.var(X)  # signal power
    sigma = np.sqrt(P_signal / 10**(BSNR / 10))  # standard deviation of the noise
    
    # Add noise
    Y = H * (Y + sigma * np.random.randn(N, N))
    
    # User-defined hyperparameters
    tau = 0.2 * sigma**2  # regularization parameter
    rho_afonso = 5e-3  # penalty parameter used by Afonso et al. (2010)
    rho = 3  # hyperparameter used in SPA
    beta = 0.2  # regularization parameter
    alpha = 1  # hyperparameter used in SPA
    
    # Number of iterations in Chambolle algorithm
    TViters = 20
    
    # MCMC parameters
    N_MC = 5000  # total number of MCMC iterations
    N_bi = 200  # number of burn-in iterations
    
    # Precomputing
    # Precompute the real matrix H for E-PO algorithm used within SPA
    Hmat = H.reshape(-1, 1)
    k2 = np.where(Hmat.flatten() == 0)[0]
    Hmat = speye(N * N, format='csr')
    
    # Remove rows corresponding to missing pixels
    mask = np.ones(N * N, dtype=bool)
    mask[k2] = False
    Hmat = Hmat[mask, :]
    
    # Precompute the real vector of observations for E-PO algorithm
    y_signal = Hmat @ X.reshape(-1, 1) + sigma * np.random.randn(Hmat.shape[0], 1)
    y_signal = y_signal.flatten()
    
    # Precompute the inverse of the precision matrix Q = 1/sigma^2 * H^T * H + 1/rho^2 * I_N
    # Using Sherman-Morrison-Woodbury formula
    identity = speye(N * N, format='csr')
    term = (rho**2) / (sigma**2 + rho**2)
    invQ_matrix = (rho**2) * (identity - term * (Hmat.T @ Hmat))
    
    # Create a function that applies invQ to a vector
    def invQ(x):
        if x.ndim == 1:
            return invQ_matrix @ x
        else:
            return invQ_matrix @ x.flatten()
    
    # Precompute the dimension of the observations vector y
    M = y_signal.shape[0]
    
    # Package all parameters
    params = {
        'X': X,
        'Y': Y,
        'N': N,
        'H': H,
        'sigma': sigma,
        'rho': rho,
        'beta': beta,
        'alpha': alpha,
        'N_MC': N_MC,
        'N_bi': N_bi,
        'Hmat': Hmat,
        'y_signal': y_signal,
        'invQ': invQ,
        'M': M
    }
    
    print('Initial parameters loaded!')
    
    return params
