"""
Chambolle's Proximal Operator for TV Regularization

Proximal point operator for the TV regularizer using Chambolle's projection algorithm.

Reference:
A. Chambolle, "An Algorithm for Total Variation Minimization and
Applications", J. Math. Imaging Vis., vol. 20, pp. 89-97, 2004.

Optimization problem:
    arg min = (1/2) || g - x ||_2^2 + lambda TV(x)
        x

Adapted by: Jose Bioucas-Dias, June 2009
Converted to Python: 2026/02/05
"""

import numpy as np


def divergence_im(p1, p2):
    """
    Compute the divergence of a 2D vector field.
    
    Parameters
    ----------
    p1 : ndarray
        First component of the vector field.
    p2 : ndarray
        Second component of the vector field.
    
    Returns
    -------
    divp : ndarray
        Divergence of the vector field.
    """
    # Horizontal component
    z = p2[:, 1:-1] - p2[:, 0:-2]
    v = np.column_stack([p2[:, 0], z, -p2[:, -1]])
    
    # Vertical component
    z = p1[1:-1, :] - p1[0:-2, :]
    u = np.vstack([p1[0, :], z, -p1[-1, :]])
    
    divp = v + u
    return divp


def gradient_im(u):
    """
    Compute the gradient of a 2D image.
    
    Parameters
    ----------
    u : ndarray
        Input image.
    
    Returns
    -------
    dux : ndarray
        Gradient in x direction.
    duy : ndarray
        Gradient in y direction.
    """
    # Gradient in x direction
    z = u[1:, :] - u[0:-1, :]
    dux = np.vstack([z, np.zeros((1, z.shape[1]))])
    
    # Gradient in y direction
    z = u[:, 1:] - u[:, 0:-1]
    duy = np.column_stack([z, np.zeros((z.shape[0], 1))])
    
    return dux, duy


def chambolle_prox_TV_stop(g, lambda_val=1, maxiter=10, tol=1e-3, tau=0.249, 
                            verbose=False, dualvars=None):
    """
    Proximal point operator for TV regularization using Chambolle's algorithm.
    
    Parameters
    ----------
    g : ndarray
        Noisy image (2D array).
    lambda_val : float, optional
        Regularization parameter (default: 1).
    maxiter : int, optional
        Maximum number of iterations (default: 10).
    tol : float, optional
        Tolerance for stopping criterion (default: 1e-3).
    tau : float, optional
        Algorithm parameter (default: 0.249).
    verbose : bool, optional
        Print iteration information (default: False).
    dualvars : tuple, optional
        Dual variables (px, py) to initialize the algorithm.
    
    Returns
    -------
    f : ndarray
        Denoised image.
    px : ndarray
        Dual variable in x direction.
    py : ndarray
        Dual variable in y direction.
    """
    
    # Initialization
    if dualvars is None:
        px = np.zeros_like(g)
        py = np.zeros_like(g)
    else:
        px, py = dualvars
    
    cont = True
    k = 0
    
    # Main iteration loop
    while cont:
        k += 1
        
        # Compute divergence of (px, py)
        divp = divergence_im(px, py)
        u = divp - g / lambda_val
        
        # Compute gradient
        upx, upy = gradient_im(u)
        tmp = np.sqrt(upx**2 + upy**2)
        
        # Compute error
        err = np.sqrt(np.sum((-upx + tmp * px)**2 + (-upy + tmp * py)**2))
        
        # Update dual variables
        px = (px + tau * upx) / (1 + tau * tmp)
        py = (py + tau * upy) / (1 + tau * tmp)
        
        # Check stopping criterion
        cont = (k < maxiter) and (err > tol)
    
    if verbose:
        print(f'\nk TV = {k}')
        print(f'err TV = {err}\n')
    
    # Compute final result
    f = g - lambda_val * divergence_im(px, py)
    
    return f, px, py
