"""
Split-and-Augmented Gibbs Sampler

This function computes the SPA algorithm to solve the linear inverse
problem y = H*x + n associated to the image inpainting problem.
"""

import numpy as np
from tqdm import tqdm
import time
import sys
sys.path.append('../utils/')
from EPO import EPO
from PMYULA import PMYULA


def SPA(y_signal, Hmat, sigma, rho, beta, alpha, N, M, invQ, N_MC):
    """
    Compute the SPA algorithm for image inpainting.
    
    Parameters
    ----------
    y_signal : ndarray
        Noisy observation (1D array).
    Hmat : sparse matrix
        Direct operator in the linear inverse problem y = H*x + n.
    sigma : float
        User-defined standard deviation of the noise.
    rho : float
        User-defined standard deviation of the variable of interest x.
    beta : float
        User-defined hyperparameter of the prior p(z).
    alpha : float
        User-defined hyperparameter of the prior p(u).
    N : int
        Dimension of X (N x N array).
    M : int
        Dimension of y (observation vector length).
    invQ : function
        Pre-computed inverse covariance matrix function.
    N_MC : int
        Total number of MCMC iterations.
    
    Returns
    -------
    X_MC : ndarray
        Samples from x (shape: N x N x N_MC).
    Z_MC : ndarray
        Samples from z (shape: N x N x N_MC).
    U_MC : ndarray
        Samples from u (shape: N x N x N_MC).
    """
    
    start_time = time.time()
    print('')
    print('BEGINNING OF THE SAMPLING')
    
    # Initialization
    # Define matrices to store the iterates
    X_MC = np.zeros((N, N, N_MC))
    Z_MC = np.zeros((N, N, N_MC))
    U_MC = np.zeros((N, N, N_MC))
    
    # Initialize the matrices
    X_MC[:, :, 0] = np.random.rand(N, N) * 255
    Z_MC[:, :, 0] = np.random.rand(N, N) * 255
    U_MC[:, :, 0] = np.random.rand(N, N) * 255
    
    # Gibbs sampling
    for t in tqdm(range(N_MC - 1), desc='Sampling in progress'):
        
        # 1. Sample x from p(x|z,u,y) using Exact Perturbation-Optimization (E-PO) method
        X_MC[:, :, t + 1] = EPO(y_signal, Hmat, sigma, U_MC[:, :, t], Z_MC[:, :, t],
                                rho, N, M, invQ)
        
        # 2. Sample z from p(z|x,u) using P-MYULA (see Durmus et al., 2018)
        Z_MC[:, :, t + 1] = PMYULA(Z_MC[:, :, t], X_MC[:, :, t + 1],
                                   U_MC[:, :, t], rho, beta, N)
        
        # 3. Sample u from p(u|x,z)
        x = X_MC[:, :, t + 1].reshape(-1)
        z = Z_MC[:, :, t + 1].reshape(-1)
        moy = (alpha**2 / (rho**2 + alpha**2)) * (z - x)
        sig = (alpha**2 * rho**2) / (alpha**2 + rho**2)
        
        # Sample from multivariate normal with diagonal covariance
        mu = moy + np.sqrt(sig) * np.random.randn(N * N)
        U_MC[:, :, t + 1] = mu.reshape(N, N)
    
    elapsed_time = time.time() - start_time
    print('END OF THE GIBBS SAMPLING')
    print(f'Execution time of the Gibbs sampling: {elapsed_time:.2f} sec')
    
    return X_MC, Z_MC, U_MC
