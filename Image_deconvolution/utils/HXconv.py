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
        The convolution kernel, for example a PSF or a gaussian kernel. This can be either the kernel (if conv is specified) or the FFT of the kernel (if conv is None).
    conv : str, optional
        Type of operation: 'Hx'. By default a standard convolution.
        If None, only returns Fourier domain matrices.
    
    Returns
    -------
    F_kernel : ndarray
        FFT of the kernel.

    conv_kernel_x : ndarray or None
        The convolved image obtained after the convolution operation with the kernel (if conv is specified).

    """
    
    m, n = x.shape
    m0, n0 = kernel.shape
    
    # Pad kernel to match size of x
    pad_pre_m = (m - m0 + 1) // 2
    pad_pre_n = (n - n0 + 1) // 2
    pad_post_m = int(np.round((m - m0 - 1) / 2))
    pad_post_n = int(np.round((n - n0 - 1) / 2))
    
    kernel_pad = np.pad(kernel, ((pad_pre_m, pad_post_m), (pad_pre_n, pad_post_n)), 
                  mode='constant', constant_values=0)
    
    # Modify the kernel to have a Fourier transform centered after the FFT
    kernel_pad = np.fft.fftshift(kernel_pad) 
    
    # Calculate the fourier transform of the kernel
    F_kernel = np.fft.fft2(kernel_pad) 
    # BCF = np.conj(BF)
    # B2F = np.abs(BF) ** 2 
    
    # Now we can calculate the convolved image
    # 1) We calculate the Fourier transform of our image x : F_x
    # 2) We multiply the two Fourier transform : F_kernel * F_x
    # 3) We calculate the image by doing the inverse Fourier transform of the product
    
    conv_kernel_x = None
    
    if conv is None:
        # return BF, BCF, B2F, y, Bpad
        return F_kernel, conv_kernel_x
    elif conv == 'Hx':
        conv_kernel_x = np.real(np.fft.ifft2(F_kernel * np.fft.fft2(x)))
        
    # elif conv == 'HTx':
    #     conv_kernel_x = np.real(np.fft.ifft2(BCF * np.fft.fft2(x)))
    # elif conv == 'HTHx':
    #     conv_kernel_x = np.real(np.fft.ifft2(B2F * np.fft.fft2(x)))
    
    # conv_kernel_x is the new image, the one obtained after the convolution with the kernel
    
    # if conv='Hx' conv_kernel_x is the convolved image
    # if conv='HTx' conv_kernel_x is the correlated, back-projected image
    # if conv='HtHx' conv_kernel_x is the doubly filtered image
    # return F_kernel, BCF, B2F, conv_kernel x, Bpad
    
    return F_kernel, conv_kernel_x
