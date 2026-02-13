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


def initialize_parameters(kernel_size=39, kernel_sigma=4, path_image='lena.bmp', gamma=6e-3,rho=20, alpha=1,delta = 1e-1, N_MC=1000, N_burn_in=200, TARGET=30, seed=1):
    """
    Initialize all parameters for the SPA algorithm.
    
    Parameters
    ----------
    kernel_size : int
        Size of the filter (size x size)
    kernel_sigma : float
        Standard deviation of the Gaussian
    path_image : str
        path of the image x
    gamma : float
        Regularization parameter.
    rho : float
        User-defined standard deviation of the variable of interest x.
    alpha : float
        User-defined hyperparameter of the prior p(u).
    N_MC : int
        Total number of MCMC iterations.
    N_burn_in : int
        Number of burn-in iterations
    TARGET : int
        Target SNR in dB
 
    Returns
    -------
    dict
        Dictionary containing all initialized parameters.
    """
    
    # Set random seed
    np.random.seed(seed)
    
    # 1.1. Load original 512 x 512 image
    # Try to load lena.bmp if available, otherwise use scikit-image's camera
    try:
        img_original = io.imread(path_image).astype(float)
        if len(img_original.shape) == 3:
            img_original = img_original[:, :, 0]  # Convert to grayscale if needed
    except:
        # Fallback to scikit-image's camera image
        print("Warning: lena.bmp not found. Using camera image instead.")
        from skimage import data
        from skimage.transform import resize
        img_original = data.camera().astype(float)
        # Resize to 512x512 if needed
        if img_original.shape[0] != 512:
            from skimage.transform import resize
            img_original = resize(img_original, (512, 512), anti_aliasing=True) * 255
    
    # 1.2. Define the regularization
    psf = np.array([[0, -1, 0],
                    [-1, 4, -1],
                    [0, -1, 0]])
    
    # FL, FLC, F2L, _, _ = HXconv(img_original, psf, 'Hx')
    # delta = 1e-1
    # FL = -FL + delta
    # FLC = -FLC + delta
    # F2L = np.abs(FL) ** 2
    
    
    F_Laplace, _= HXconv(img_original, psf, 'Hx') # smooth prior
    F_Laplace = -F_Laplace + delta
        
    
    
    # 1.3. Define the blurring kernel and its associated Fourier matrices
    blur_kernel = fspecial_gaussian(kernel_size, kernel_sigma)
    # FB, FBC, F2B, Bx, _ = HXconv(img_original, B, 'Hx')
    
    F_blur_kernel, conv_blur_kernel_x = HXconv(img_original, blur_kernel , 'Hx')
    
    
    # 1.4. Apply the blurring operator on the original image
    
    N_pixel = img_original.size
    Ni = int(np.sqrt(N_pixel))
    
    target_SNR = TARGET  # in dB
    signal_power=np.mean(conv_blur_kernel_x**2)
    
    # calculate std of noise to achieve target SNR
    noise_power = signal_power / (10**(target_SNR / 10))
    # On peut faire directement signal_power=np.mean pour enlever le M1
    noise_std = np.sqrt(noise_power)
    
    # Changer par rng pour reproductibilité du code
    # rng = np.random.default_rng(seed=seed) # TO DO
    # noise_samples = noise_std * rng.standard_normal((Ni, Ni))
    D = noise_std # D le bruit
    img_noisy = conv_blur_kernel_x + D * np.random.randn(Ni, Ni)
    
    # # 1.5. Define the parameters of SPA
    # rho = rho
    # alpha = alpha
    
    # On peut techniquement enlever le 1.6
    # 1.6. Define MCMC parameters 
    N_MC = N_MC  # total number of MCMC iterations
    N_burn_in = N_burn_in   # number of burn-in iterations
    
    # 1.7. Other parameters and precomputing
    # gamma = 6e-3  # regularization parameter (fixed here)
    # D = D ** (-2)  # precision matrix associated to the likelihood
    # mu1 = 0.99 / np.max(D)  # parameter used in AuxV1 method embedded in SPA
    # N = y.shape[0]
    
    # 1.7. Other parameters and precomputing
    if target_SNR <= 25:
        gamma = 1e-2  # Plus de régularisation pour SNR 20

    D = D**(-2) # precision matrix associated to the likelihood
    mu1 = 0.99 / D # parameter used in AuxV1 method embedded in SPA
    N = img_noisy.shape[0] # Number of lines or column of y
    
    # Package all parameters
    params = {
        'D': D, # Precision matrix
        'mu1': mu1, # parameter used in AuxV1 method embedded in SPA
        'F_blur_kernel': F_blur_kernel, # H
        'F_Laplace': F_Laplace, # L
        # 'F2B': F2B, # kerne H^T*H=|H|^2
        'rho': rho,
        'alpha': alpha,
        'img_noisy': img_noisy,
        #'FBC': FBC,
        'gamma': gamma,
        #'F2L': F2L, # H^T  and 
        'N': N,
        'N_MC': N_MC,
        'N_burn_in': N_burn_in,
        'img_original': img_original
    }
    
    print('Initial parameters loaded!')
    
    return params
