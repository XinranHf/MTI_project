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
    
    
    return F_kernel, conv_kernel_x
