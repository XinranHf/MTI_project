import numpy as np
from tqdm import tqdm

def AuxV1_sampler(y, FB, F2L, params):
    """
    Implémentation de l'algorithme AuxV1.
    Très efficace pour la déconvolution avec bruit Gaussien.
    """
    N, N_MC = params['N'], params['N_MC']
    X_MC = np.zeros((N, N, N_MC))
    D = params['D']
    mu1 = params['mu1']
    gamma = params['gamma']
    
    # Initialisation
    X_MC[:, :, 0] = np.random.rand(N, N)
    
    for t in tqdm(range(N_MC - 1), desc="AuxV1 Sampling"):
        # 1. Échantillonnage de la variable auxiliaire v 
        Hx = np.real(np.fft.ifft2(FB * np.fft.fft2(X_MC[:, :, t])))
        moy_v = (1/mu1 - D) * Hx
        sigma_v = np.sqrt(1/mu1 - D)
        v = moy_v + sigma_v * np.random.randn(N, N)
        
        # 2. Échantillonnage de x (Fourier)
        # Précision = (1/mu1) * |H|^2 + gamma * |L|^2
        prec_x = (1/mu1) * np.abs(FB)**2 + gamma * F2L
        rhs_x = np.conj(FB) * np.fft.fft2(D * y + v)
        
        eps = (np.random.randn(N, N) + 1j * np.random.randn(N, N)) * np.sqrt(0.5)
        FX = rhs_x / prec_x + eps / np.sqrt(prec_x)
        X_MC[:, :, t+1] = np.real(np.fft.ifft2(FX))
        
    return X_MC

def AuxV2_sampler(y, FB, F2L, params):
    """
    Implémentation de l'algorithme AuxV2.
    Nécessite généralement 3x plus d'itérations que SPA/AuxV1. 
    """
    N, N_MC = params['N'], params['N_MC']
    X_MC = np.zeros((N, N, N_MC))
    D = params['D']
    gamma = params['gamma']
    
    # Paramètre mu2 spécifique pour assurer la convergence [cite: 714]
    mu2 = 0.99 / np.max(D)
    
    X_MC[:, :, 0] = np.random.rand(N, N)
    
    for t in tqdm(range(N_MC - 1), desc="AuxV2 Sampling"):
        # 1. Échantillonnage de v (spatial)
        Hx = np.real(np.fft.ifft2(FB * np.fft.fft2(X_MC[:, :, t])))
        mean_v = (1 - mu2 * D) * Hx + mu2 * D * y
        std_v = np.sqrt(mu2 * (1 - mu2 * D))
        v = mean_v + std_v * np.random.randn(N, N)
        
        # 2. Échantillonnage de x (Fourier)
        prec_x = np.abs(FB)**2 + mu2 * gamma * F2L
        rhs_x = np.conj(FB) * np.fft.fft2(v)
        
        eps = (np.random.randn(N, N) + 1j * np.random.randn(N, N)) * np.sqrt(0.5)
        FX = rhs_x / prec_x + eps * (np.sqrt(mu2) / np.sqrt(prec_x))
        X_MC[:, :, t+1] = np.real(np.fft.ifft2(FX))
        
    return X_MC