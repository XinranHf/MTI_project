"""
HXconv function for convolution operations in Fourier domain.

Author: Ningning Zhao (University of Toulouse) - Original MATLAB version
Date: 2015/03/28
Converted to Python: 2026/02/05

Note: 
    - x is the RF signal or isreal(x)=1
    - B is shift invariant and circular boundary is considered
"""

import numpy as np


def HXconv(x, kernel, conv=None):
    """
    Compute convolution operations using Fourier transforms.
    
    Parameters
    ----------
    x : ndarray
        Input signal/image (2D array).
    kernel : ndarray
        Either the PSF (if conv is specified) or the FFT of PSF (if conv is None).
    conv : str, optional
        Type of operation: 'Hx'.
        If None, only returns Fourier domain matrices.
    
    Returns
    -------
    F_kernel : ndarray
        FFT of the padded PSF.

    conv_kernel_x : ndarray or None
        Result of convolution operation (if conv is specified).

    """
    
    m, n = x.shape
    m0, n0 = kernel.shape
    
    # Pad kernel to match size of x
    # Equivalent to MATLAB's: padarray(kernel, floor([m-m0+1, n-n0+1]/2), 'pre')
    #                   then: padarray(kernel_pad, round([m-m0-1, n-n0-1]/2), 'post')
    pad_pre_m = (m - m0 + 1) // 2
    pad_pre_n = (n - n0 + 1) // 2
    pad_post_m = int(np.round((m - m0 - 1) / 2))
    pad_post_n = int(np.round((n - n0 - 1) / 2))
    
    kernel_pad = np.pad(kernel, ((pad_pre_m, pad_post_m), (pad_pre_n, pad_post_n)), 
                  mode='constant', constant_values=0)
    kernel_pad = np.fft.fftshift(kernel_pad) # Modify the kernel to have a Fourier transform centered after the FFT
    
    F_kernel = np.fft.fft2(kernel_pad) # Calculates H, H^T and H^T*H=|H|^2 at the same time to reduce computational resourcesH
    # BCF = np.conj(BF)
    # B2F = np.abs(BF) ** 2 
    
    conv_kernel_x = None
    
    if conv is None:
        # return BF, BCF, B2F, y, Bpad
        return F_kernel, conv_kernel_x
    elif conv == 'Hx':
        conv_kernel_x = np.real(np.fft.ifft2(F_kernel * np.fft.fft2(x)))
    # elif conv == 'HTx':
    #     y = np.real(np.fft.ifft2(BCF * np.fft.fft2(x)))
    # elif conv == 'HTHx':
    #     y = np.real(np.fft.ifft2(B2F * np.fft.fft2(x)))
    
    # y is the new image 
    # if conv='Hx' y is the convolved image
    # if conv='HTx' y is the correlated, back-projected image
    # if conv='HtHx' y is the doubly filtered image
    # return BF, BCF, B2F, y, Bpad
    
    return F_kernel, conv_kernel_x
