"""
E-PO Algorithm to Sample the Variable X

This function computes the E-PO algorithm as described in the paper of C.
Gilavert et al., 2015. This algorithm deals with the exact resolution
case of the linear system Q*x = eta and with a guaranteed convergence to
the target distribution.
"""

import numpy as np
from scipy.sparse import eye as speye


def EPO(y, H, sigma, U, Z, rho, N, M, invQ):
    """
    Compute the E-PO algorithm for sampling x.
    
    Parameters
    ----------
    y : ndarray
        Noisy observation (1D array).
    H : sparse matrix
        Direct operator in the linear inverse problem y = H*x + n.
    sigma : float
        User-defined standard deviation of the noise.
    U : ndarray
        Current MCMC iterate of U (2D array).
    Z : ndarray
        Current MCMC iterate of Z (2D array).
    rho : float
        User-defined standard deviation of the variable of interest x.
    N : int
        Dimension of X (N x N array).
    M : int
        Dimension of y (observation vector length).
    invQ : function
        Pre-computed inverse covariance matrix function.
    
    Returns
    -------
    x : ndarray
        Sample from the posterior distribution of x (2D array).
    """
    
    # 1. Sample eta from N(Q*mu, Q)
    
    # 1.1. Sample eta_y from N(y, sigma^2 * I_M)
    eta_y = y + sigma * np.random.randn(M)
    
    # 1.2. Sample eta_x from N(z - u, rho^2 * I_N)
    u = U.reshape(-1)
    z = Z.reshape(-1)
    eta_x = (z - u) + rho * np.random.randn(N * N)
    
    # 1.3. Set eta
    eta_aux = (1 / sigma**2) * (H.T @ eta_y) + (1 / rho**2) * eta_x
    
    # 2. Compute an exact solution x_new of Q*x = eta <=> x = invQ*eta
    x = invQ(eta_aux)
    x = x.reshape(N, N)
    
    return x
