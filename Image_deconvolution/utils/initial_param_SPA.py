"""
Initial variables and parameters for SPA algorithm.

This module generates and saves initial parameters for the SPA algorithm
applied to image deconvolution.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import data, io
from HXconv import HXconv
import os


def fspecial_gaussian(size, sigma):
    """
    Create a Gaussian filter similar to MATLAB's fspecial('gaussian').
    
    Parameters
    ----------
    size : int
        Size of the filter (size x size).
    sigma : float
        Standard deviation of the Gaussian.
    
    Returns
    -------
    ndarray
        Gaussian filter kernel.
    """
    m = n = size
    h, k = m // 2, n // 2
    x, y = np.mgrid[-h:h+1, -k:k+1]
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return g / g.sum()


def initialize_parameters():
    """
    Initialize all parameters for the SPA algorithm.
    
    Returns
    -------
    dict
        Dictionary containing all initialized parameters.
    """
    
    # Set random seed
    np.random.seed(1)
    
    # 1.1. Load original 512 x 512 image
    # Try to load lena.bmp if available, otherwise use scikit-image's camera
    try:
        refl = io.imread('lena.bmp').astype(float)
        if len(refl.shape) == 3:
            refl = refl[:, :, 0]  # Convert to grayscale if needed
    except:
        # Fallback to scikit-image's camera image
        print("Warning: lena.bmp not found. Using camera image instead.")
        from skimage import data
        from skimage.transform import resize
        refl = data.camera().astype(float)
        # Resize to 512x512 if needed
        if refl.shape[0] != 512:
            from skimage.transform import resize
            refl = resize(refl, (512, 512), anti_aliasing=True) * 255
    
    # 1.2. Define the regularization
    psf = np.array([[0, -1, 0],
                    [-1, 4, -1],
                    [0, -1, 0]])
    
    FL, FLC, F2L, _, _ = HXconv(refl, psf, 'Hx')
    delta = 1e-1
    FL = -FL + delta
    FLC = -FLC + delta
    F2L = np.abs(FL) ** 2
    
    # 1.3. Define the blurring kernel and its associated Fourier matrices
    B = fspecial_gaussian(39, 4)
    FB, FBC, F2B, Bx, _ = HXconv(refl, B, 'Hx')
    
    # 1.4. Apply the blurring operator on the original image
    
    #N = refl.size
    #Ni = int(np.sqrt(N))
    #beta = 0.35
    #kappa1 = 13
    #kappa2 = 40
    
    #D = kappa2 * np.random.binomial(1, beta, (Ni, Ni)).astype(float)
    #D[D == 0] = kappa1
    #y = Bx + D * np.random.randn(Ni, Ni)
    
    ####
    
    N = refl.size
    Ni = int(np.sqrt(N))
    
    target_SNR = 30  # in dB
    #signal_power = np.sum(Bx**2)
    signal_power=np.mean(Bx**2)
    
    #M1, M2 = B.shape
    #noise_power = signal_power / (M1*10**(target_SNR / 10))
    
    # calculate std of noise to achieve target SNR
    noise_power = signal_power / (10**(target_SNR / 10))
    # On peut faire directement signal_power=np.mean pour enlever le M1
    noise_std = np.sqrt(noise_power)
    
    # Changer par rng pour reproductibilité du code
    # rng = np.random.default_rng(1)
    # noise_samples = noise_std * rng.standard_normal((Ni, Ni))
    D = noise_std * np.ones((Ni, Ni)) # D le bruit
    y = Bx + D * np.random.randn(Ni, Ni)
    
    # 1.5. Define the parameters of SPA
    rho = 20
    alpha = 1
    
    # 1.6. Define MCMC parameters
    N_MC = 1000  # total number of MCMC iterations
    N_bi = 200   # number of burn-in iterations
    
    # 1.7. Other parameters and precomputing
    # gamma = 6e-3  # regularization parameter (fixed here)
    # D = D ** (-2)  # precision matrix associated to the likelihood
    # mu1 = 0.99 / np.max(D)  # parameter used in AuxV1 method embedded in SPA
    # N = y.shape[0]
    
    # 1.7. Other parameters and precomputing
    if target_SNR <= 25:
        gamma = 1e-2  # Plus de régularisation pour SNR 20
    else:
        gamma = 6e-3  # regularization parameter (fixed here)
        
    #D = D ** (-2)  # precision matrix associated to the likelihood
    # We recalculate the precision of the matrix
    # Les 2 D ont une utilités différentes ?
    precision = 1.0 / (noise_std**2)
    D = np.ones((Ni, Ni)) * precision
    mu1 = 0.99 / np.max(D)  # parameter used in AuxV1 method embedded in SPA
    N = y.shape[0]
    
    # Package all parameters
    params = {
        'D': D,
        'mu1': mu1,
        'FB': FB,
        'F2B': F2B,
        'rho': rho,
        'alpha': alpha,
        'y': y,
        'FBC': FBC,
        'gamma': gamma,
        'F2L': F2L,
        'N': N,
        'N_MC': N_MC,
        'N_bi': N_bi,
        'refl': refl
    }
    
    print('Initial parameters loaded!')
    
    return params
