# -*- coding: utf-8 -*-
from param import Param
from grid import Grid
from fluid2d import Fluid2d
import numpy as np
import matplotlib.pyplot as plt

param = Param('default.xml')
param.modelname = 'boussinesq'
param.expname = 'khi_0'

# domain and resolution
ratio = 1
param.ny = 2**7
param.nx = param.ny*ratio
param.Ly = 1.
param.Lx = 1*ratio
param.npx = 1
param.npy = 1
param.geometry = 'xchannel'

# time
param.tend = 2.
param.cfl = 1.5
param.adaptable_dt = True
param.dt = 0.01
param.dtmax = 0.02

# discretization
param.order = 5
param.timestepping = 'RK3_SSP'

# output
param.var_to_save = ['vorticity', 'psi', 'u', 'v', 'buoyancy', 'banom']
param.list_diag = 'all'
param.freq_plot = 5
param.freq_his = .25
param.freq_diag = .1

# plot
param.plot_interactive = True
param.plot_var = 'buoyancy'
param.cax = [-8., 8.]
param.colorscheme = 'imposed'
param.generate_mp4 = True
param.cmap = 'inferno'

# physics
param.forcing = False
param.noslip = False
param.diffusion = False
param.forcing = False

param.gravity = 1.

nh = param.nh

grid = Grid(param)


f2d = Fluid2d(param, grid)
model = f2d.model

xr, yr = grid.xr, grid.yr
vor = model.var.get('vorticity')
buoy = model.var.get('buoyancy')

# control parameters of the experiment
N = 4.  # Brunt Vaisala frequency squared
S = 20.  # vertical shear


def set_buoyancy(param, grid, x0, y0, sigma,
           density_type, ratio=1, sharpness = 100):
    

    xr, yr = grid.xr, grid.yr
    # ratio controls the ellipticity, ratio=1 is a disc
    x = np.sqrt((xr-param.Lx*x0)**2+(yr-param.Ly*y0)**2*ratio**2)

    y = x.copy()*0.

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



# linear stratification
buoy[:, :] = 0.
dtype = 'smooth_step'
sigma = 0.3*param.Lx

buoy[:] += set_buoyancy(param, grid, 0.5, 0.5, sigma,
                dtype, ratio=1, sharpness = 400)*-2



def get_phi(buoy):
    phi =  np.amax(buoy)-buoy
    return phi/(np.amax(phi)-np.amin(phi))

def gradient(phi):
    """Compute the gradient of phi with differents approches: in directions i,
    j and along the diagonals.
    The returned gradient is averaged between the two methods.
    A second averaging is done with adjacent cells."""
    
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
    grad_i[1:-1,1:-1] = phi[2:,1:-1]-phi[:-2,1:-1]
    grad_j[1:-1,1:-1] = phi[1:-1,2:]-phi[1:-1,:-2]

    # Diagonal gradients:
    sqrt2 = np.sqrt(2)
    grad_dj[1:-1,1:-1] = (phi[:-2,2:]-phi[2:,:-2])/sqrt2
    grad_di[1:-1,1:-1] = (phi[2:,2:]-phi[:-2,:-2])/sqrt2    
    
    # Projecting diagonal gradients, onto the i and i directions,
    # adding to i and j gradients and averaging:
    grad_j[1:-1,1:-1] = (grad_j[1:-1,1:-1]+(
        grad_dj[1:-1,1:-1]+grad_di[1:-1,1:-1])/sqrt2)/2 
    
    grad_i[1:-1,1:-1] = (grad_i[1:-1,1:-1]+(
        -grad_dj[1:-1,1:-1]+grad_di[1:-1,1:-1])/sqrt2)/2
    
    # Averaging with adjacent cells:
    grad_i = average(grad_i)
    grad_j = average(grad_j)
    return -grad_i, -grad_j


def normal(v_i, v_j):
    norm = np.sqrt(v_i**2+v_j**2)
    norm[norm == 0] = 1
    return v_i/norm, v_j/norm

model.bref[:] = buoy

print('Ri = %4.2f' % (N**2/S**2))

vor[:, :] = 0.


grad_i, grad_j = gradient(get_phi(buoy))
n_i, n_j = normal( *gradient(get_phi(buoy)))
grad = np.sqrt(grad_i**2 + grad_j**2)

# In order to check the look the initial conditions.
plt.figure(figsize = (10,10))

#plt.pcolormesh(grid.xr, grid.yr, get_phi(buoy), shading = 'nearest')
plt.pcolormesh(grid.xr, grid.yr, grad**4, shading = 'nearest')

n_i[grad<0.6] = 0
n_j[grad<0.6] = 0
#plt.quiver(grid.xr, grid.yr, n_j, n_i, scale = 10, minlength = 0)

plt.yticks([])
plt.xticks([])
plt.show()
#model.set_psi_from_vorticity()

#f2d.loop()

