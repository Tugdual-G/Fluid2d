# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 14:30:15 2021

@author: tugdual
"""
import numpy as np
from numba import jit, prange

@jit(nopython = True, parallel=True)
def average(n, weight = 0.15):
    sqrt2 = np.sqrt(2)
    """return the weighted average with adjacent cells."""        
    n[1:-1, 1:-1] = (n[2:,2:]/sqrt2 + n[2:,1:-1] + n[2:,:-2]/sqrt2 +
                    n[:-2,:-2]/sqrt2 + n[:-2, 1:-1] + n[:-2,2:]/sqrt2 +
                    n[1:-1,:-2] + n[1:-1, 2:])*(1-weight)/8 + n[1:-1, 1:-1]*weight


@jit(nopython = True, parallel=True)
def gradient(phi, dx):
    """Compute the gradient of phi with differents approches: in directions i,
    j and along the diagonals.
    The returned gradient is averaged between the two methods.
    A second averaging is done with adjacent cells."""
       
    # Creating gradients arrays:
    grad_i = np.zeros_like(phi, dtype = np.float32)
    grad_j = np.zeros_like(phi, dtype = np.float32)
    """grad_di = np.zeros_like(phi)
    grad_dj = np.zeros_like(phi)"""
    
    # Computing the gradient in direction i(vertical) and j(horizontal):
    grad_i[1:-1,1:-1] = (phi[2:,1:-1]-phi[:-2,1:-1])/(2*dx)
    grad_j[1:-1,1:-1] = (phi[1:-1,2:]-phi[1:-1,:-2])/(2*dx)
    
    # Boudaries:
    grad_i[0,1:-1] = (phi[1,1:-1]-phi[0,1:-1])/dx
    grad_i[-1,1:-1] = (phi[-1,1:-1]-phi[-2,1:-1])/dx       
    grad_i[1:-1,0] = (phi[2:,0]-phi[:-2,0])/(2*dx)    
    grad_i[1:-1,-1] = (phi[2:,-1]-phi[:-2,-1])/(2*dx)     
    # Boudaries:
    grad_j[0,1:-1] = (phi[0,2:]-phi[0,:-2])/(2*dx)    
    grad_j[-1,1:-1] = (phi[-1,2:]-phi[-1,:-2])/(2*dx)      
    grad_j[1:-1, 0] = (phi[1:-1,1]-phi[1:-1,0])/dx    
    grad_j[1:-1, -1] = (phi[1:-1,-1]-phi[1:-1,-2])/dx        
    
    """# Diagonal gradients:
    sqrt2 = np.sqrt(2)
    grad_dj[1:-1,1:-1] = (phi[:-2,2:]-phi[2:,:-2])/(2*sqrt2*dx)
    grad_di[1:-1,1:-1] = (phi[2:,2:]-phi[:-2,:-2])/(2*sqrt2*dx)    
    
    # Projecting diagonal gradients, onto the i and i directions,
    # adding to i and j gradients and averaging:
    grad_j[1:-1,1:-1] = (grad_j[1:-1,1:-1]+(
        grad_dj[1:-1,1:-1]+grad_di[1:-1,1:-1])/sqrt2)/2 
    
    grad_i[1:-1,1:-1] = (grad_i[1:-1,1:-1]+(
        -grad_dj[1:-1,1:-1]+grad_di[1:-1,1:-1])/sqrt2)/2"""
    
    # Averaging with adjacent cells:
    #average(grad_i)
    #average(grad_j)
    return grad_i, grad_j

@jit(nopython = True, parallel=True)
def laplacian(x,dx):
    lplc = np.zeros_like(x, dtype = np.float64)
    lplc[1:-1,1:-1] = (x[:-2,1:-1] + x[2:,1:-1] + x[1:-1,:-2] + x[1:-1,2:]
                     - 4*x[1:-1,1:-1])/(dx**2)
    #boundaries:
    #!!! Ne fonctionne pas pour le moment
    # left 
    lplc[1:-1,0] = (x[2:,0] + x[:-2,0]-2*x[1:-1,0])/(dx**2) + 2*(x[1:-1,2]-2*x[1:-1,1]-x[1:-1,0])/(3*dx**2)
    # Right 
    lplc[1:-1,-1] = (x[2:,-1] + x[:-2,-1]-2*x[1:-1,-1])/(dx**2) + 2*(x[1:-1,-3]-2*x[1:-1,-2]+x[1:-1,-1])/(3*dx**2)
    # Bottom
    lplc[0,1:-1] = 2*(x[2,1:-1]-2*x[1,1:-1]-x[0,1:-1])/(3*dx**2) + (x[0,2:] + x[0,:-2]-2*x[0,1:-1])/(dx**2)
    # Top
    lplc[-1,1:-1] = 2*(x[-3,1:-1]-2*x[-2,1:-1]+x[-1,1:-1])/(3*dx**2) + (x[-1,2:] + x[-1,:-2]-2*x[-1,1:-1])/(dx**2)  
    
    return lplc

@jit(nopython = True, parallel=True)
def divergence(x_i, x_j, dx):
    div = np.zeros_like(x_i, dtype = np.float64)
    m, n = np.shape(x_i)
    div[1:-1,1:-1] = ((x_i[2:,1:-1]-x_i[:-2,1:-1]) + (x_j[1:-1,2:]-
                                                   x_j[1:-1,:-2]))/(2*dx)
    
    #boundaries
    div[0,1:-1] = (x_i[1,1:-1]-x_i[0,1:-1])/dx + (x_i[0,2:]-x_i[0,:-2])/(2*dx)
    div[-1,1:-1] = (x_i[-1,1:-1]-x_i[-2,1:-1])/dx + (x_i[-1,2:]-x_i[-1,:-2])/(2*dx)    
    #boundaries
    div[1:-1,0] = (x_i[2:,0]-x_i[:-2,0])/(2*dx) + (x_i[1:-1,1]-x_i[1:-1,0])/dx
    div[1:-1,-1] = (x_i[2:,-1]-x_i[:-2,-1])/(2*dx) + (x_i[1:-1,-1]-x_i[1:-1,-2])/dx
    return div



@jit(nopython = True, parallel=True)
def normalise(v_i, v_j):
    norm = np.empty_like(v_i)
    for i in prange(np.shape(v_i)[0]):
        for j in prange(np.shape(v_i)[1]):            
            norm[i,j] = np.sqrt(v_i[i,j]**2+v_j[i,j]**2)
            if norm[i,j] == 0:
                norm[i,j] = 1
    return -v_i/norm, -v_j/norm

@jit(nopython = True, parallel=True)
def curl(x_i,x_j, dx):
    crl = np.zeros_like(x_i, dtype = np.float64)
    crl[1:-1,1:-1] = ((x_i[1:-1,2:]-x_i[1:-1,:-2])-(x_j[2:,1:-1]-x_j[:-2,1:-1]))/(2*dx)
    
    #boundaries
    crl[0,1:-1] = ((x_i[0,2:]-x_i[0,:-2])/(2*dx)-(x_j[1,1:-1]-x_j[0,1:-1]))/dx
    crl[-1,1:-1] = ((x_i[-1,2:]-x_i[-1,:-2])/(2*dx)-(x_j[-1,1:-1]-x_j[-2,1:-1]))/dx
    #boundaries
    crl[1:-1,0] = ((x_i[1:-1,1]-x_i[1:-1,0])/dx-(x_j[2:,0]-x_j[:-2,0]))/(2*dx)
    crl[1:-1,-1] = ((x_i[1:-1,-2]-x_i[1:-1,-1])/dx-(x_j[2:,-1]-x_j[:-2,-1]))/(2*dx)   
    
    return crl
    
@jit(nopython = True, parallel=True)
def source_1d(dxdt, phi, dx, xi, M):
    """return the source term in the 1d equation, 
    conservation of mass, using phi"""
 
    grd_phi_i, grd_phi_j = gradient(phi, dx)
    
    n_i, n_j = normalise(grd_phi_i, grd_phi_j)
    
    dxdt += divergence(M*(grd_phi_i-4/xi*phi*(1-phi)*n_i),
                           M*(grd_phi_j-4/xi*phi*(1-phi)*n_j), dx)

@jit(nopython = True, parallel=True)
def torque(dxdt, phi, dx, rho_l, rho_h, xi, sigma, gravity=10):
    """Return the torque term in vorticity equation"""    
    kappa = 3/2*sigma*xi
    eta = 12*sigma/xi
 
    grd_phi_i, grd_phi_j = gradient(phi, dx)    
        
    F_i = ((1-phi)*rho_l + phi*rho_h)*gravity + (4*eta*phi*(phi-1)*(phi-1/2)-
                         kappa*laplacian(phi,dx))*grd_phi_i
    
    F_j = (4*eta*phi*(phi-1)*(phi-1/2)-kappa*laplacian(phi,dx))*grd_phi_j
    
    dxdt += curl(F_i, F_j, dx)
    
    # In order to maintain phi in [0,1]:
    for i in prange(np.shape(phi)[0]):
        for j in prange(np.shape(phi)[1]):
            if phi[i,j] > 1:
                phi[i,j] = 1
            if phi[i,j] < 0:
                phi[i,j] = 0

@jit(nopython = True, parallel=True)
def viscosity(dxdt, w, dx, phi, nu_l = 15*10**-6, nu_h = 10**-5):
    dxdt += laplacian(w,dx)*((1-phi)*nu_l + phi*nu_h)
        