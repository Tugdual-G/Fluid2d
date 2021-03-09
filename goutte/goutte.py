# -*- coding: utf-8 -*-
from param import Param
from grid import Grid
from fluid2d import Fluid2d
import numpy as np
import matplotlib.pyplot as plt
from droplet_operator import gradient, normalise, torque, source_1d

param = Param('default.xml')
param.modelname = 'droplet'
param.expname = 'khi_0'

# domain and resolution
ratio = 0.5
param.ny = 2**9
param.nx = param.ny*ratio
param.Ly = 1.
param.Lx = 1*ratio
param.npx = 1
param.npy = 1
param.geometry = 'closed'

# time
param.tend = 0.4
param.cfl = 1.5
param.adaptable_dt = True
param.dt = 0.001
param.dtmax = 0.01

# discretization
param.order = 5
param.timestepping = 'RK3_SSP'

# output
param.var_to_save = ['phi']
param.list_diag = 'all'
param.freq_plot = 5
param.freq_his = .01
param.freq_diag = .1

# plot
param.plot_interactive = False
param.plot_var = 'phi'
param.cax = [0., 1.]
param.colorscheme = 'imposed'
param.generate_mp4 = True
param.cmap = 'viridis'

# physics
param.forcing = False
param.noslip = True
param.diffusion = False
param.forcing = False

param.gravity = 1.

nh = param.nh

grid = Grid(param)


f2d = Fluid2d(param, grid)
model = f2d.model



def set_buoyancy(param, grid, x0, y0, sigma,
           density_type, ratio=1, sharpness = 200):
    

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

def get_phi(buoy):
    phi =  np.amax(buoy)-buoy
    return phi/(np.amax(phi)-np.amin(phi))

def add_phi(phi, param, grid, x0, y0, sigma, ratio=1, sharpness = 100):   
    xr, yr = grid.xr, grid.yr
    # ratio controls the ellipticity, ratio=1 is a disc
    r = np.sqrt((xr-param.Lx*x0)**2+(yr-param.Ly*y0)**2*ratio**2)

    y = np.abs(1-np.tanh((r-sigma)*sharpness))
    y /= np.amax(y)
    phi[phi<y] = y[phi<y]

def del_phi(phi, param, grid, x0, y0, sigma, ratio=1, sharpness = 100):
    xr, yr = grid.xr, grid.yr
    # ratio controls the ellipticity, ratio=1 is a disc
    r = np.sqrt((xr-param.Lx*x0)**2+(yr-param.Ly*y0)**2*ratio**2)

    y = np.abs(1+np.tanh((r-sigma)*sharpness))
    y /= np.amax(y)
    phi[phi>y] = y[phi>y]

vor = model.var.get('vorticity')
phi = model.var.get('phi')




# Creating a distribution of high density liquid: 



phi[:,:] = 0.
h = 0.2
phi[grid.yr<h+0.015]= 0.25
phi[grid.yr<h+0.01]= 0.5
phi[grid.yr<h+0.005]= 0.75
phi[grid.yr<h] = 1.

# Droplets:
add_phi(phi, param, grid, 0.5, 0.88, 0.03)




vor[:, :] = 0.


# =============================================================================
# In order to check the look of the initial conditions.
# =============================================================================


#dx = param.Ly/param.ny
#grad_i, grad_j = gradient(phi, dx)
#n_i, n_j = normalise( grad_i, grad_j)
#grad = np.sqrt(grad_i**2 + grad_j**2)

#computing the torque:
#trq = np.zeros_like(phi)
#torque(trq, phi, dx, rho_l = 1, rho_h = 10, xi = 5*dx, sigma = 0.01)

#plt.figure(figsize = (8,8))

# Check phi:
#plt.pcolormesh(grid.xr, grid.yr, phi, shading = 'nearest')
#plt.colorbar()

#Check the torque:
#plt.pcolormesh(grid.xr, grid.yr, trq, shading = 'nearest')

# Check the source term in equation (1d) :
#plt.pcolormesh(grid.xr, grid.yr, source_1d(phi, dx, xi = 3*dx, M = 0.01), shading = 'nearest')


#Check the gradient:
#grad_i[grad<30] = 0
#grad_j[grad<30] = 0
#plt.quiver(grid.xr, grid.yr, -grad_j, -grad_i, scale = 500, minlength = 0)


model.set_psi_from_vorticity()

f2d.loop()

"""plt.figure(figsize = (8,8))
plt.pcolormesh(grid.xr, grid.yr, phi, shading = 'nearest')
plt.colorbar()"""
