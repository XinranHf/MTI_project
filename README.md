# Split-and-Augmented Gibbs Sampler (SPA)  
## Traitement du signal Course Project - SDI Centrale Lille 

This repository contains a **Python reimplementation and reproducibility study** of the Split-and-Augmented Gibbs Sampler (SPA) algorithm for large-scale Bayesian inverse problems.

This work was carried out as part of a university course project.  
The objective was to:

- Reproduce the methodology presented in the original article
- Translate the MATLAB implementation into Python
- Validate the algorithm on image deconvolution and inpainting tasks


## Reference Article

M. Vono et al., *"Split-and-augmented Gibbs sampler - Application to large-scale inference problems"*, 2018.

This repository is student reproduction for academic purposes as part of a course project and may differ slightly from the original MATLAB code.


# Project Objectives

- Understand the theoretical foundations of SPA
- Implement the algorithm in Python
- Reproduce numerical experiments
- Analyze convergence and reconstruction quality
- Study the influence of noise and regularization parameters


# Problem Setting

We consider the inverse problem:

$$
y = Hx + \varepsilon
$$

where:

- $x$ is the unknown image
- $H$ is a blur operator (Gaussian convolution)
- $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$ is Gaussian noise

Application studied: Image deconvolution  


# Noise Management and SNR Control

A key aspect of this reproduction work is the **controlled generation of noise**.

Instead of choosing noise arbitrarily, we fix a **target Signal-to-Noise Ratio (SNR)**:

```python
TARGET = 20  # dB
````

The noise variance is computed as:

$$
\sigma^2 = \frac{\text{signal power}}{10^{\text{SNR}/10}}
$$

### Procedure:

1. Blur the original image
2. Compute signal power:
   `mean((Hx)^2)`
3. Deduce the required noise variance
4. Generate Gaussian noise accordingly

This ensures:

* Reproducible experiments
* Fair comparison across different SNR levels
* Controlled degradation severity

For low SNR values (≤ 25 dB), the regularization parameter `gamma` is automatically increased to improve stability.


# Implementation Details

## FFT-based Convolution

All convolution operations are performed in the Fourier domain for efficiency using NumPy FFT.

## Prior Model

A Laplacian smoothness prior is used:

```python
psf = [[0, -1, 0],
       [-1, 4, -1],
       [0, -1, 0]]
```

Hyperparameters:

* `kernel_size` (default: 39) – Size of the Gaussian blur kernel (39×39)
* `kernel_sigma` (default: 4) – Standard deviation of the Gaussian blur
* `gamma` (default: 6e-3) – Regularization strength
* `rho` (default: 20) – Prior variance parameter
* `alpha` (default: 1) – Auxiliary variable hyperparameter
* `delta` (default: 1e-1) – Small constant for numerical stability in the Laplacian operator
* `N_MC` (default: 1000) – Total number of MCMC iterations
* `N_burn_in` (default: 200) – Number of burn-in iterations
* `TARGET` (default: 20 dB) – Target Signal-to-Noise Ratio used to generate noise
* `seed` (default: 1) – Random seed for reproducibility


## Reproducibility

* Explicit random seed control via `numpy.random.default_rng`
* Automatic fallback image if `lena.bmp` is not available
* Deterministic SNR-based noise generation


# Outputs

The experiments produce:

* PSNR and SNR metrics
* MMSE reconstruction
* 90% credibility intervals
* Visualization of:
  * Original image
  * Blurred + noisy observation
  * Restored image

# Project Structure

```
Image_deconvolution/
├── experiments/
│   ├── SPA_lena.py
├── src/
│   ├── SPA.py
└── utils/
    ├── HXconv.py
    ├── initial_parameters.py
```
