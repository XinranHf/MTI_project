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


def HXconv(x, B, conv=None):
    """
    Compute convolution operations using Fourier transforms.
    
    Parameters
    ----------
    x : ndarray
        Input signal/image (2D array).
    B : ndarray
        Either the PSF (if conv is specified) or the FFT of PSF (if conv is None).
    conv : str, optional
        Type of operation: 'Hx', 'HTx', or 'HTHx'.
        If None, only returns Fourier domain matrices.
    
    Returns
    -------
    BF : ndarray
        FFT of the padded PSF.
    BCF : ndarray
        Complex conjugate of BF.
    B2F : ndarray
        Magnitude squared of BF.
    y : ndarray or None
        Result of convolution operation (if conv is specified).
    Bpad : ndarray
        Padded PSF.
    """
    
    m, n = x.shape
    m0, n0 = B.shape
    
    # Pad B to match size of x
    # Equivalent to MATLAB's: padarray(B, floor([m-m0+1, n-n0+1]/2), 'pre')
    #                   then: padarray(Bpad, round([m-m0-1, n-n0-1]/2), 'post')
    pad_pre_m = (m - m0 + 1) // 2
    pad_pre_n = (n - n0 + 1) // 2
    pad_post_m = int(np.round((m - m0 - 1) / 2))
    pad_post_n = int(np.round((n - n0 - 1) / 2))
    
    Bpad = np.pad(B, ((pad_pre_m, pad_post_m), (pad_pre_n, pad_post_n)), 
                  mode='constant', constant_values=0)
    Bpad = np.fft.fftshift(Bpad)
    
    BF = np.fft.fft2(Bpad)
    BCF = np.conj(BF)
    B2F = np.abs(BF) ** 2
    
    y = None
    
    if conv is None:
        return BF, BCF, B2F, y, Bpad
    elif conv == 'Hx':
        y = np.real(np.fft.ifft2(BF * np.fft.fft2(x)))
    elif conv == 'HTx':
        y = np.real(np.fft.ifft2(BCF * np.fft.fft2(x)))
    elif conv == 'HTHx':
        y = np.real(np.fft.ifft2(B2F * np.fft.fft2(x)))
    
    return BF, BCF, B2F, y, Bpad
