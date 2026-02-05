"""
Sampling the Splitting Variable Z

This function samples the splitting variable Z thanks to a proximal MCMC
algorithm called P-MYULA (see Durmus et al., 2018).
"""

import numpy as np
from chambolle_prox_TV_stop import chambolle_prox_TV_stop


def PMYULA(Z, X, U, rho, beta, N):
    """
    Sample the splitting variable Z using P-MYULA algorithm.
    
    Parameters
    ----------
    Z : ndarray
        Current MCMC iterate of Z (2D array).
    X : ndarray
        Current MCMC iterate of X (2D array).
    U : ndarray
        Current MCMC iterate of U (2D array).
    rho : float
        User-defined standard deviation of the variable of interest x.
    beta : float
        User-defined hyperparameter in p(z).
    N : int
        Dimension of X (N x N array).
    
    Returns
    -------
    Z_new : ndarray
        New value for Z (2D array).
    """
    
    # Pre-processing
    u = U.reshape(-1)
    x = X.reshape(-1)
    z = Z.reshape(-1)
    lambda_MYULA = rho**2  # as prescribed in Durmus et al.
    gamma_MYULA = (rho**2) / 4  # as prescribed in Durmus et al.
    
    # 1. Sample the zero-mean Gaussian variable b
    b = np.random.randn(N * N)
    
    # 2. Update the value of Z
    
    # 2.1. Compute the gradient of f(z) = (1 / (2 * rho^2)) * ||z - (x + u)||_2^2
    grad_f = (1 / rho**2) * (z - (x + u))
    
    # 2.2. Compute the proximal operator of g: prox(z)^(lambda_MYULA)_g
    prox_z, _, _ = chambolle_prox_TV_stop(Z, lambda_val=beta * lambda_MYULA, maxiter=20)
    prox_z = prox_z.reshape(-1)
    
    # 2.3. Compute the new value of z: z_new
    z_new = ((1 - gamma_MYULA / lambda_MYULA) * z -
             gamma_MYULA * grad_f +
             (gamma_MYULA / lambda_MYULA) * prox_z +
             np.sqrt(2 * lambda_MYULA) * b)
    
    # 2.4. Reshape z_new (1D-array) into Z_new (2D-array)
    Z_new = z_new.reshape(N, N)
    
    return Z_new
