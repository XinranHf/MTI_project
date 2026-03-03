"""
Split-and-Augmented Gibbs Sampler

This function computes the SPA algorithm to solve the linear inverse
problem y = H*x + n associated to the image deconvolution problem.
"""

import numpy as np
from tqdm import tqdm
import time


def SPA(D, mu1, F_blur_kernel, rho, alpha, img_noisy, gamma, F_Laplace, N, N_MC):
    """
    Compute the SPA algorithm for image deconvolution.
    
    Parameters
    ----------
    D : ndarray
        Precision matrix associated to the likelihood.
    mu1 : float
        Hyperparameter used in the AuxV1 algorithm.
    F_blur_kernel : ndarray
        Counterpart of the blur operator in the Fourier domain.
    rho : float
        User-defined standard deviation of the variable of interest x.
    alpha : float
        User-defined hyperparameter of the prior p(u).
    img_noisy : ndarray
        Observations (2D-array).
    gamma : float
        Regularization parameter.
    F_Laplace : ndarray
        Counterpart of the Laplace operator in the Fourier domain.
    N : int
        Dimension of x (N x N array).
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
    # print('BEGINNING OF THE SAMPLING')
    
    # Initialization
    # Define matrices to store the iterates
    X_MC = np.zeros((N, N, N_MC))
    Z_MC = np.zeros((N, N, N_MC))
    U_MC = np.zeros((N, N, N_MC))
    
    # Initialize the matrices
    X_MC[:, :, 0] = np.random.rand(N, N) * 255
    Z_MC[:, :, 0] = np.random.rand(N, N) * 255
    U_MC[:, :, 0] = np.random.rand(N, N) * 255
    
    # Precompute Fourier domain matrices
    F_conj_blur_kernel = np.conj(F_blur_kernel)
    F_abs_blur_kernel = np.abs(F_blur_kernel) ** 2
    F_abs_Laplace = np.abs(F_Laplace) ** 2
     
    
    # Gibbs sampling
    for t in tqdm(range(N_MC - 1), desc='Sampling in progress'):
        
        # Sampling x with the method AuxV1
        
        # Sampling the auxiliary variable v1
        moy = (1 / mu1 - D) * np.real(np.fft.ifft2(F_blur_kernel * np.fft.fft2(X_MC[:, :, t])))
        moy = moy.reshape(-1)
        sigma = 1 / mu1 - D.reshape(-1)
        
        # Sample from multivariate normal (independent components due to diagonal covariance)
        v1 = moy + np.sqrt(sigma) * np.random.randn(N * N)
        v1 = v1.reshape(N, N)
        
        # Sampling the variable of interest x
        z0 = np.fft.fft2(Z_MC[:, :, t])
        u0 = np.fft.fft2(U_MC[:, :, t])
        precision = (1 / mu1) * F_abs_blur_kernel + (1 / rho**2)
        moy = (F_conj_blur_kernel * np.fft.fft2(D * img_noisy) +
               (1 / rho**2) * (z0 - u0) +
               F_conj_blur_kernel * np.fft.fft2(v1)) / precision
        
        eps = np.sqrt(0.5) * (np.random.randn(N, N) + 1j * np.random.randn(N, N))
        x0 = moy + eps / np.sqrt(precision)
        X_MC[:, :, t + 1] = np.real(np.fft.ifft2(x0))
        
        # Sampling z
        precision = gamma * F_abs_Laplace.reshape(-1) + (1 / rho**2)
        x0 = np.fft.fft2(X_MC[:, :, t + 1]).reshape(-1)
        u0 = np.fft.fft2(U_MC[:, :, t]).reshape(-1)
        moy = (1 / rho**2) * (x0 + u0) / precision
        
        eps = np.sqrt(0.5) * (np.random.randn(N * N) + 1j * np.random.randn(N * N))
        z0 = moy + eps / np.sqrt(precision)
        Z_MC[:, :, t + 1] = np.real(np.fft.ifft2(z0.reshape(N, N)))
        
        # Sampling u
        cov = (alpha**2 * rho**2) / (alpha**2 + rho**2)
        moy = np.fft.fft2(Z_MC[:, :, t + 1] - X_MC[:, :, t + 1]) * alpha**2 / (rho**2 + alpha**2)
        moy = moy.reshape(-1)
        
        eps = np.sqrt(0.5) * (np.random.randn(N * N) + 1j * np.random.randn(N * N))
        u0 = moy + eps * np.sqrt(cov)
        U_MC[:, :, t + 1] = np.real(np.fft.ifft2(u0.reshape(N, N)))
    
    elapsed_time = time.time() - start_time
    # print('END OF THE GIBBS SAMPLING')
    print(f'Execution time of the Gibbs sampling: {elapsed_time:.2f} sec')
    
    return X_MC, Z_MC, U_MC, elapsed_time