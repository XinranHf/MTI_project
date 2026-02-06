# Split-and-Augmented Gibbs Sampler (SPA) - Python Version

This is a Python conversion of the MATLAB implementation of the Split-and-Augmented Gibbs Sampler applied to image deconvolution and image inpainting.

## Reference

M. VONO et al., "Split-and-augmented Gibbs sampler - Application to large-scale inference problems", submitted, 2018.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Image Deconvolution

Navigate to the experiments directory and run:

```bash
cd Image_deconvolution/experiments
python SPA_lena.py
```

### Image Inpainting

Navigate to the experiments directory and run:

```bash
cd Image_inpainting/experiments
python SPA_cameraman.py
```

## Project Structure

```
Image_deconvolution/
├── experiments/
│   ├── SPA_lena.py          # Main script for Lena image deconvolution
│   └── SPA_lena.m           # Original MATLAB version
├── src/
│   ├── SPA.py               # SPA algorithm implementation
│   └── SPA.m                # Original MATLAB version
└── utils/
    ├── HXconv.py            # Convolution utilities
    ├── initial_param_
   - Deconvolution: Tries to load `lena.bmp` but falls back to scikit-image's camera image if not found.
   - Inpainting: Tries to load `cameraman.tif` but falls back to scikit-image's camera image if not found.

3. **FFT**: Uses NumPy's FFT implementation (`numpy.fft.fft2` and `numpy.fft.ifft2`).

4. **Sparse Matrices**: Uses SciPy's sparse matrix implementation (`scipy.sparse`) instead of MATLAB's sparse matrices.

5. **Multivariate Normal Sampling**: MATLAB's `mvnrnd` is replaced with direct sampling using NumPy for diagonal covariance matrices.

6. **Progress Bar**: Uses `tqdm` instead of MATLAB's `waitbar`.

7. **Plotting**: Uses Matplotlib instead of MATLAB's plotting functions.

## Parameters

### Image Deconvolution
- **N_MC**: Total number of MCMC iterations (default: 1000)
- **N_bi**: Number of burn-in iterations (default: 200)
- **rho**: Standard deviation parameter (default: 20)
- **alpha**: Prior hyperparameter (default: 1)
- **gamma**: Regularization parameter (default: 6e-3)

### Image Inpainting
- **N_MC**: Total number of MCMC iterations (default: 5000)
- **N_bi**: Number of burn-in iterations (default: 200)
- **rho**: Standard deviation parameter (default: 3)
- **alpha**: Prior hyperparameter (default: 1)
- **beta**: Regularization parameter (default: 0.2)
- **BSNR**: Blurred Signal-to-Noise Ratio (default: 40 dB)

## Output

### Image Deconvolution
The algorithm produces:
- PSNR and SNR metrics
- Visualizations of:
  - Original image
  - Blurred and noisy observation
  - MMSE estimates of x, z, and u
  - 90% credibility intervals

### Image Inpainting
The algorithm produces:
- ISNR, MSE, and SSIM metrics
- Visualizations of:
  - Original image
  - Decimat
2. **Image Loading**: The Python version tries to load `lena.bmp` but falls back to scikit-image's camera image if not found.

3. **FFT**: Uses NumPy's FFT implementation (`numpy.fft.fft2` and `numpy.fft.ifft2`).

4. **Multivariate Normal Sampling**: MATLAB's `mvnrnd` is replaced with direct sampling using NumPy for diagonal covariance matrices.

5. **Progress Bar**: Uses `tqdm` instead of MATLAB's `waitbar`.
**Image Files**: 
  - For deconvolution: Have `lena.bmp` in the working directory, or the code will use a fallback image.
  - For inpainting: Have `cameraman.tif` in the working directory, or the code will use scikit-image's camera.
- **Computational Intensity**: 
  - Deconvolution: ~1000 iterations, takes a few minutes
  - Inpainting: ~5000 iterations, more computationally intensive, may take 10-30 minutes
- **Numerical Differences**: Results may differ slightly from MATLAB due to differences in random number generation and numerical precision.
- **Memory Requirements**: The inpainting algorithm stores large arrays of samples and may require significant RAM for large images
## Parameters

- **N_MC**: Total number of MCMC iterations (default: 1000)
- **N_bi**: Number of burn-in iterations (default: 200)
- **rho**: Standard deviation parameter (default: 20)
- **alpha**: Prior hyperparameter (default: 1)
- **gamma**: Regularization parameter (default: 6e-3)

## Output

The algorithm produces:
- PSNR and SNR metrics
- Visualizations of:
  - Original image
  - Blurred and noisy observation
  - MMSE estimates of x, z, and u
  - 90% credibility intervals

## Notes

- Make sure you have the `lena.bmp` file in the appropriate directory, or the code will use a fallback image.
- The algorithm is computationally intensive and may take several minutes to complete.
- Results may differ slightly from MATLAB due to differences in random number generation and numerical precision.
