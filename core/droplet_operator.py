# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 14:30:15 2021

@author: tugdual
"""
import numpy as np
from numba import jit


def gradient(phi, dx):
    """Compute the gradient of phi with differents approches: in directions i,
    j and along the diagonals.
    The returned gradient is averaged between the two methods.
    A second averaging is done with adjacent cells."""

    def average(n, weight = 0.15):
        sqrt2 = np.sqrt(2, dtype = np.float32)
        """return the weighted average with adjacent cells."""        
        n[1:-1, 1:-1] = (n[2:,2:]/sqrt2 + n[2:,1:-1] + n[2:,:-2]/sqrt2 +
                        n[:-2,:-2]/sqrt2 + n[:-2, 1:-1] + n[:-2,2:]/sqrt2 +
                        n[1:-1,:-2] + n[1:-1, 2:])*(1-weight)/8 + n[1:-1, 1:-1]*weight
        return n

    # Creating gradients arrays:
    grad_i = np.zeros_like(phi, dtype = np.float32)
    grad_j = np.zeros_like(phi, dtype = np.float32)
    grad_di = np.zeros_like(phi)
    grad_dj = np.zeros_like(phi)
    
    # Computing the gradient in direction i(vertical) and j(horizontal):
    grad_i[1:-1,1:-1] = (phi[2:,1:-1]-phi[:-2,1:-1])/(2*dx)
    grad_j[1:-1,1:-1] = (phi[1:-1,2:]-phi[1:-1,:-2])/(2*dx)

    # Diagonal gradients:
    sqrt2 = np.sqrt(2)
    grad_dj[1:-1,1:-1] = (phi[:-2,2:]-phi[2:,:-2])/(2*sqrt2*dx)
    grad_di[1:-1,1:-1] = (phi[2:,2:]-phi[:-2,:-2])/(2*sqrt2*dx)    
    
    # Projecting diagonal gradients, onto the i and i directions,
    # adding to i and j gradients and averaging:
    grad_j[1:-1,1:-1] = (grad_j[1:-1,1:-1]+(
        grad_dj[1:-1,1:-1]+grad_di[1:-1,1:-1])/sqrt2)/2 
    
    grad_i[1:-1,1:-1] = (grad_i[1:-1,1:-1]+(
        -grad_dj[1:-1,1:-1]+grad_di[1:-1,1:-1])/sqrt2)/2
    
    # Averaging with adjacent cells:
    #grad_i = average(grad_i)
    #grad_j = average(grad_j)
    return grad_i, grad_j


def laplacian(x,dx):
    lplc = np.zeros_like(x, dtype = np.float32)
    lplc[1:-1,1:-1] = (x[:-2,1:-1] + x[2:,1:-1] + x[1:-1,:-2] + x[1:-1,2:]
                     - 4*x[1:-1,1:-1])/(dx**2)
    return lplc


def divergence(x_i, x_j, dx):
    div = np.zeros_like(x_i, dtype = np.float32)
    div[1:-1,1:-1] = ((x_i[2:,1:-1]-x_i[:-2,1:-1])+(x_j[1:-1,2:]-
                                                   x_j[1:-1,:-2]))/(2*dx)
    return div


def normalise(v_i, v_j):
    norm = np.sqrt(v_i**2+v_j**2, dtype = np.float32)
    norm[norm == 0] = 1
    return -v_i/norm, -v_j/norm


def curl(x_i,x_j, dx):
    crl = np.zeros_like(x_i, dtype = np.float32)
    crl[1:-1,1:-1] = ((x_i[1:-1,2:]-x_i[1:-1,:-2])-(x_j[2:,1:-1]-x_j[:-2,1:-1]))/(2*dx)
    return crl
    

def source_1d(dxdt, phi, dx, xi, M):
    """return the source term in the 1d equation, 
    conservation of mass, using phi"""
       
    grd_phi_i, grd_phi_j = gradient(phi, dx)
    
    n_i, n_j = normalise(grd_phi_i, grd_phi_j)
    
    dxdt += divergence(M*(grd_phi_i-4/xi*phi*(1-phi)*n_i),
                           M*(grd_phi_j-4/xi*phi*(1-phi)*n_j), dx)


def torque(dxdt, phi, dx, rho_l, rho_h, xi, sigma, gravity=10):
    """Return the torque term in vorticity equation"""    
    kappa = 3/2*sigma*xi
    eta = 12*sigma/xi
    rho = (1-phi)*rho_l + phi*rho_h    
    
    grd_phi_i, grd_phi_j = gradient(phi, dx)    
        
    F_i = rho*gravity + (4*eta*phi*(phi-1)*(phi-1/2)-
                         kappa*laplacian(phi,dx))*grd_phi_i
    
    F_j = (4*eta*phi*(phi-1)*(phi-1/2)-kappa*laplacian(phi,dx))*grd_phi_j
    
    dxdt += curl(F_i, F_j, dx)
    
    
    
    
    
    
    
    
    
    
    
    
    