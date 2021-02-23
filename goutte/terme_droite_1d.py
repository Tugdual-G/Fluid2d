#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 09:32:29 2021

@author: tugdual
"""
import numpy as np
import matplotlib.pyplot as plt

nx = 2**7
ny = nx
Lx = 1
Ly = 1
dx = Lx/nx
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X,Y = np.meshgrid(x,y)


def set_buoyancy(X, Y, Lx, Ly,  x0, y0, sigma,
           density_type, ratio=1, sharpness = 200):
    

    # ratio controls the ellipticity, ratio=1 is a disc
    x = np.sqrt((X-Lx*x0)**2+(Y-Ly*y0)**2*ratio**2)

    y = x*0.

    if density_type in ('gaussian', 'cosine', 'step', 'smooth_step'):
        if density_type == 'gaussian':
            y = np.exp(-x**2/(sigma**2))

        elif density_type == 'cosine':
            y = np.cos(x/sigma*np.pi/2)
            y[x > sigma] = 0.

        elif density_type == 'step':
            y[x <= sigma] = 1.
        
        elif density_type == 'smooth_step':
            y = -np.tanh((x-sigma)*sharpness)
    else:
        print('this kind of density (%s) is not defined' % density_type)

    return y



def get_phi(buoy):
    phi =  np.amax(buoy)-buoy
    return phi/(np.amax(phi)-np.amin(phi))

def gradient(nx, Lx, phi):
    """Compute the gradient of phi with differents approches: in directions i,
    j and along the diagonals.
    The returned gradient is averaged between the two methods.
    A second averaging is done with adjacent cells."""
    dx = Lx/nx
    def average(n, weight = 0.15):
        sqrt2 = np.sqrt(2)
        """return the weighted average with adjacent cells."""        
        n[1:-1, 1:-1] = (n[2:,2:]/sqrt2 + n[2:,1:-1] + n[2:,:-2]/sqrt2 +
                        n[:-2,:-2]/sqrt2 + n[:-2, 1:-1] + n[:-2,2:]/sqrt2 +
                        n[1:-1,:-2] + n[1:-1, 2:])*(1-weight)/8 + n[1:-1, 1:-1]*weight
        return n

    # Creating gradients arrays:
    grad_i = phi*0
    grad_j = phi*0
    grad_di = phi*0
    grad_dj = phi*0
    
    # Computing the gradient in direction i(vertical) and j(horizontal):
    grad_i[1:-1,1:-1] = (phi[2:,1:-1]-phi[:-2,1:-1])/(2*dx)
    grad_j[1:-1,1:-1] = (phi[1:-1,2:]-phi[1:-1,:-2])/(2*dx)

    # Diagonal gradients:
    sqrt2 = np.sqrt(2)
    grad_dj[1:-1,1:-1] = (phi[:-2,2:]-phi[2:,:-2])/(sqrt2*dx*2)
    grad_di[1:-1,1:-1] = (phi[2:,2:]-phi[:-2,:-2])/(sqrt2*dx*2)    
    
    # Projecting diagonal gradients, onto the i and i directions,
    # adding to i and j gradients and averaging:
    grad_j[1:-1,1:-1] = (grad_j[1:-1,1:-1]+(
        grad_dj[1:-1,1:-1]+grad_di[1:-1,1:-1])/sqrt2)/2 
    
    grad_i[1:-1,1:-1] = (grad_i[1:-1,1:-1]+(
        -grad_dj[1:-1,1:-1]+grad_di[1:-1,1:-1])/sqrt2)/2
    
    # Averaging with adjacent cells:
    grad_i = average(grad_i)
    grad_j = average(grad_j)
    return grad_i, grad_j


def normal(v_i, v_j):
    norm = np.sqrt(v_i**2+v_j**2)
    norm[norm == 0] = 1
    return -v_i/norm, -v_j/norm

def deriv(Lx, nx, f):
    dx = Lx/nx
    grad_i = f*0
    grad_j = f*0
    grad_i[1:-1,1:-1] = (f[2:,1:-1]-f[:-2,1:-1])/(2*dx)
    grad_j[1:-1,1:-1] = (f[1:-1,2:]-f[1:-1,:-2])/(2*dx)
    return grad_i, grad_j




# Création d'une goutte
buoy = np.zeros_like(X)
dtype = 'smooth_step'
sigma = 0.2*Lx

buoy[:] = set_buoyancy(X, Y, Lx, Ly, 0.5, 0.5, sigma,
                dtype, ratio=3, sharpness = 50)*-2

# Récupération de la fonction indicatrice
phi = get_phi(buoy)
grad_i, grad_j = gradient(nx, Lx, phi)
n_i, n_j = normal( *gradient(nx, Lx, phi))
grad = np.sqrt(grad_i**2 + grad_j**2)

"""
grad_i[grad<30] = 0
grad_j[grad<30] = 0
plt.quiver(X, Y, -grad_j, -grad_i, scale = 200, minlength = 0)"""


# =============================================================================
# On observe ce que fait le terme de l'équation (1d)
# =============================================================================
    
plt.figure(figsize = (7,7))
plt.title('t={}'.format(0))  
plt.pcolormesh(X, Y, grad, shading = 'nearest')
plt.show()
tend = 0.15
dt = 0.01
n_step = int(tend/dt)+1
t = 0
ksi = 2*dx
M = 0.001
phi = get_phi(buoy)

for i in range(n_step):
    grad_i, grad_j = gradient(nx, Lx, phi)
    n_i, n_j = normal( grad_i, grad_j)    
    term_i = M*(grad_i-4/ksi*phi*(1-phi)*n_i)
    term_j = M*(grad_j-4/ksi*phi*(1-phi)*n_j)
    dphi = (deriv(Lx, nx, term_i)[0]+deriv(Lx, nx, term_j)[1])*dt
    phi += dphi
    t += dt
    if i%6 == 0:
        plt.figure(figsize = (7,7))
        grad = np.sqrt(grad_i**2 + grad_j**2)
        plt.title('t={}'.format(round(t,3)))
        plt.pcolormesh(X, Y, grad, shading = 'nearest')
        plt.show()
    
 
