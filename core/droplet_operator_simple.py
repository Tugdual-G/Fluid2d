# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 14:30:15 2021

@author: tugdual
"""
import numpy as np
from numba import jit, prange


@jit(nopython=True, parallel=True, cache=True)
def gradient_i(phi, dx):
    """Compute the gradient of phi with differents approches: in directions i,
    j and along the diagonals.
    The returned gradient is averaged between the two methods.
    A second averaging is done with adjacent cells."""

    # Creating gradients arrays:
    grad_i = np.zeros_like(phi, dtype=np.float32)
    grad_di = np.zeros_like(phi)
    grad_dj = np.zeros_like(phi)

    """# Computing the gradient in direction i(vertical) and j(horizontal):
    m, n = np.shape(phi)
    for i in prange(1, m-1):
        # Boudaries:
        grad_i[0, i] = (phi[1, i]-phi[0, i])/dx
        grad_i[-1, i] = (phi[-1, i]-phi[-2, i])/dx
        grad_i[i, 0] = (phi[i+1, 0]-phi[i-1, 0])/(2*dx)
        grad_i[i, -1] = (phi[i+1, -1]-phi[i-1, -1])/(2*dx)
        for j in prange(1, n-1):
            grad_i[i, j] = (phi[i+1, j]-phi[i-1, j])/(2*dx)"""

    # Diagonal gradients:
    sqrt2 = np.sqrt(2)
    grad_dj[1:-1, 1:-1] = (phi[:-2, 2:]-phi[2:, :-2])/(2*sqrt2*dx)
    grad_di[1:-1, 1:-1] = (phi[2:, 2:]-phi[:-2, :-2])/(2*sqrt2*dx)

    # Projecting diagonal gradients, onto the i and i directions,
    # adding to i and j gradients and averaging:

    """grad_i[1:-1, 1:-1] = (grad_i[1:-1, 1:-1]+(
        -grad_dj[1:-1, 1:-1]+grad_di[1:-1, 1:-1])/sqrt2)/2"""
    grad_i[1:-1, 1:-1] = (-grad_dj[1:-1, 1:-1]+grad_di[1:-1, 1:-1])/sqrt2
    return grad_i


@jit(nopython=True, parallel=True, cache=True)
def gradient_j(phi, dx):
    """Compute the gradient of phi with differents approches: in directions i,
    j and along the diagonals.
    The returned gradient is averaged between the two methods.
    A second averaging is done with adjacent cells."""

    # Creating gradients arrays:
    grad_j = np.zeros_like(phi, dtype=np.float32)
    grad_di = np.zeros_like(phi)
    grad_dj = np.zeros_like(phi)

    """# Computing the gradient in direction i(vertical) and j(horizontal):
    m, n = np.shape(phi)
    for i in prange(1, m-1):
        # Boudaries:
        grad_j[0, i] = (phi[0, i+1]-phi[0, i-1])/(2*dx)
        grad_j[-1, i] = (phi[-1, i+1]-phi[-1, i-1])/(2*dx)
        grad_j[i, 0] = (phi[i, 1]-phi[i, 0])/dx
        grad_j[i, -1] = (phi[i, -1]-phi[i, -2])/dx
        for j in prange(1, n-1):
            grad_j[i, j] = (phi[i, j+1]-phi[i, j-1])/(2*dx)"""

    # Diagonal gradients:
    sqrt2 = np.sqrt(2)
    grad_dj[1:-1, 1:-1] = (phi[:-2, 2:]-phi[2:, :-2])/(2*sqrt2*dx)
    grad_di[1:-1, 1:-1] = (phi[2:, 2:]-phi[:-2, :-2])/(2*sqrt2*dx)

    # Projecting diagonal gradients, onto the i and i directions,
    # adding to i and j gradients and averaging:
    """grad_j[1:-1, 1:-1] = (grad_j[1:-1, 1:-1]+(
        grad_dj[1:-1, 1:-1]+grad_di[1:-1, 1:-1])/sqrt2)/2"""
    grad_j[1:-1, 1:-1] = (grad_dj[1:-1, 1:-1]+grad_di[1:-1, 1:-1])/sqrt2

    return grad_j


@jit(nopython=True, parallel=True, cache=True)
def sec_deriv_i(y, dx):
    der = np.zeros_like(y)
    m, n = np.shape(y)
    for i in prange(1, m-1):
        for j in prange(n):
            der[i, j] = (y[i-1, j] + y[i+1, j] - 2*y[i, j])/(dx**2)
    return der


@jit(nopython=True, parallel=True, cache=True)
def sec_deriv_j(y, dx):
    der = np.zeros_like(y)
    m, n = np.shape(y)
    for i in prange(m):
        for j in prange(1, n-1):
            der[i, j] = (y[i, j-1] + y[i, j+1] - 2*y[i, j])/(dx**2)
    return der


@jit(nopython=True, parallel=True, cache=True)
def third_deriv_i(y, dx):
    der = np.zeros_like(y)
    m, n = np.shape(y)
    for i in prange(2, m-2):
        for j in prange(n):
            der[i, j] = (y[i+2, j] - y[i-2, j] - 2 *
                         (y[i+1, j]-y[i-1, j]))/(2*dx**3)
    return der


@jit(nopython=True, parallel=True, cache=True)
def third_deriv_j(y, dx):
    der = np.zeros_like(y)
    m, n = np.shape(y)
    for i in prange(m):
        for j in prange(2, n-2):
            der[i, j] = (y[i, j+2] - y[i, j-2] - 2 *
                         (y[i, j+1]-y[i, j-1]))/(2*dx**3)
    return der


@jit(nopython=True, parallel=True, cache=True)
def laplacian(x, dx):
    lplc = np.zeros_like(x)
    m, n = np.shape(x)
    for i in prange(1, m-1):
        for j in prange(1, n-1):
            lplc[i, j] = (x[i-1, j] + x[i+1, j] + x[i, j-1] +
                          x[i, j+1]-4*x[i, j])/(dx**2)

    # boundaries:
    #!!! Pas certain du calcul
    # left
    lplc[1:-1, 0] = (x[2:, 0] + x[:-2, 0]-2*x[1:-1, 0]) / \
        (dx**2) + 2*(x[1:-1, 2]-2*x[1:-1, 1]-x[1:-1, 0])/(3*dx**2)
    # Right
    lplc[1:-1, -1] = (x[2:, -1] + x[:-2, -1]-2*x[1:-1, -1]) / \
        (dx**2) + 2*(x[1:-1, -3]-2*x[1:-1, -2]+x[1:-1, -1])/(3*dx**2)
    # Bottom
    lplc[0, 1:-1] = 2*(x[2, 1:-1]-2*x[1, 1:-1]-x[0, 1:-1]) / \
        (3*dx**2) + (x[0, 2:] + x[0, :-2]-2*x[0, 1:-1])/(dx**2)
    # Top
    lplc[-1, 1:-1] = 2*(x[-3, 1:-1]-2*x[-2, 1:-1]+x[-1, 1:-1]) / \
        (3*dx**2) + (x[-1, 2:] + x[-1, :-2]-2*x[-1, 1:-1])/(dx**2)

    return lplc


@jit(nopython=True, parallel=True, cache=True)
def divergence(x_i, x_j, dx):
    div = np.zeros_like(x_i, dtype=np.float64)
    m, n = np.shape(x_i)
    for i in prange(1, m-1):
        for j in prange(1, n-1):
            div[i, j] = ((x_i[i+1, j]-x_i[i-1, j]) +
                         (x_j[i, j+1]-x_j[i, j-1]))/(2*dx)

    # boundaries
    div[0, 1:-1] = (x_i[1, 1:-1]-x_i[0, 1:-1])/dx + \
        (x_i[0, 2:]-x_i[0, :-2])/(2*dx)
    div[-1, 1:-1] = (x_i[-1, 1:-1]-x_i[-2, 1:-1])/dx + \
        (x_i[-1, 2:]-x_i[-1, :-2])/(2*dx)
    # boundaries
    div[1:-1, 0] = (x_i[2:, 0]-x_i[:-2, 0])/(2*dx) + \
        (x_i[1:-1, 1]-x_i[1:-1, 0])/dx
    div[1:-1, -1] = (x_i[2:, -1]-x_i[:-2, -1])/(2*dx) + \
        (x_i[1:-1, -1]-x_i[1:-1, -2])/dx
    return div


@jit(nopython=True, parallel=True, cache=True)
def normalise(v_i, v_j):
    norm = np.empty_like(v_i)
    for i in prange(np.shape(v_i)[0]):
        for j in prange(np.shape(v_i)[1]):
            norm[i, j] = np.sqrt(v_i[i, j]**2+v_j[i, j]**2)
            if norm[i, j] == 0:
                norm[i, j] = 1

    return -v_i/norm, -v_j/norm


@jit(nopython=True, parallel=True, cache=True)
def curl(x_i, x_j, dx):
    crl = np.zeros_like(x_i, dtype=np.float64)
    m, n = np.shape(x_i)
    for i in prange(1, m-1):
        for j in prange(1, n-1):
            crl[i, j] = ((x_i[i, j+1]-x_i[i, j-1]) -
                         (x_j[i+1, j]-x_j[i-1, j]))/(2*dx)

    # boundaries
    crl[0, 1:-1] = ((x_i[0, 2:]-x_i[0, :-2])/(2*dx) -
                    (x_j[1, 1:-1]-x_j[0, 1:-1]))/dx
    crl[-1, 1:-1] = ((x_i[-1, 2:]-x_i[-1, :-2])/(2*dx) -
                     (x_j[-1, 1:-1]-x_j[-2, 1:-1]))/dx
    # boundaries
    crl[1:-1, 0] = ((x_i[1:-1, 1]-x_i[1:-1, 0])/dx -
                    (x_j[2:, 0]-x_j[:-2, 0]))/(2*dx)
    crl[1:-1, -1] = ((x_i[1:-1, -2]-x_i[1:-1, -1])/dx -
                     (x_j[2:, -1]-x_j[:-2, -1]))/(2*dx)

    return crl


@jit(nopython=True, parallel=True, cache=True)
def source_1d(dxdt, phi, dx, xi, M):
    """return the source term in the 1d equation, 
    conservation of mass, using phi"""

    grd_phi_i, grd_phi_j = gradient_i(phi, dx), gradient_j(phi, dx)

    n_i, n_j = normalise(grd_phi_i, grd_phi_j)

    dxdt += divergence(M*(grd_phi_i-4/xi*phi*(1-phi)*n_i),
                       M*(grd_phi_j-4/xi*phi*(1-phi)*n_j), dx)


@jit(nopython=True, parallel=True, cache=True)
def torque(dxdt, phi, density, dx, rho_l, rho_h, xi, sigma, gravity=10):
    """Return the torque term in vorticity equation"""
    kappa = 3/2*sigma*xi
    eta = 12*sigma/xi
    density[:, :] = ((1-phi)*rho_l + phi*rho_h)

    kap_i = -kappa*(third_deriv_i(phi, dx) +
                    gradient_i(sec_deriv_j(phi, dx), dx))
    kap_j = -kappa*(third_deriv_j(phi, dx) +
                    gradient_j(sec_deriv_i(phi, dx), dx))

    crochet_i = gradient_i(4*eta*phi*(phi-1)*(phi-1/2), dx)+kap_i
    crochet_j = gradient_j(4*eta*phi*(phi-1)*(phi-1/2), dx)+kap_j

    dxdt += gradient_j(density, dx)*gravity + crochet_j * \
        gradient_i(phi, dx)-crochet_i*gradient_j(phi, dx)


@jit(nopython=True, parallel=True, cache=True)
def restrict_phi(phi):
    """Ensure that phi is in [0,1]"""
    for i in prange(np.shape(phi)[0]):
        for j in prange(np.shape(phi)[1]):
            if phi[i, j] > 1:
                phi[i, j] = 1.
            if phi[i, j] < 0:
                phi[i, j] = 0.


@jit(nopython=True, parallel=True, cache=True)
def viscosity(dxdt, w, dx, phi, nu_l=15*10**-6, nu_h=10**-3):
    dxdt += laplacian(w, dx)*((1-phi)*nu_l + phi*nu_h)


@jit(nopython=True, parallel=True, cache=True)
def anti_diffusion(phi, nc=2):
    m, n = np.shape(phi)
    for i in prange(nc, m-nc):
        for j in prange(nc, n-nc):
            if phi[i, j] > 0 and phi[i, j] < 1:
                if((phi[i, j]-0.5)*(phi[i-nc, j]-0.5) > 0 and
                   (phi[i, j]-0.5)*(phi[i+nc, j]-0.5) > 0 and
                   (phi[i, j]-0.5)*(phi[i, j-nc]-0.5) > 0 and
                   (phi[i, j]-0.5)*(phi[i, j+nc]-0.5) > 0):

                    if (phi[i, j] > 0.5):
                        phi[i, j] = 1
                    if (phi[i, j] < 0.5):
                        phi[i, j] = 0
