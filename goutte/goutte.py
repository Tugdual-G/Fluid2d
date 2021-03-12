# -*- coding: utf-8 -*-
from param import Param
from grid import Grid
from fluid2d import Fluid2d
import numpy as np
import matplotlib.pyplot as plt


param = Param('default.xml')
param.modelname = 'droplet'
param.expname = 'drop_diag'

# domain and resolution
ratio = 1
param.ny = 2**7
param.nx = param.ny*ratio
param.Ly = 1.
param.Lx = param.Ly*ratio
param.npx = 1
param.npy = 1
param.geometry = 'closed'

# time
param.tend = 0.2
param.cfl = 1.5
param.adaptable_dt = True
param.dt = 0.001
param.dtmax = 0.01

# discretization
param.order = 5
param.timestepping = 'RK4_LS'

# output
param.var_to_save = ['phi', "u", "v", "vorticity"]
param.list_diag = 'all'
param.freq_plot = 5
param.freq_his = .005
param.freq_diag = .1

# plot
param.plot_interactive = True
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

# rho, sigma and M
param.rho_h = 1000.
param.rho_l = 1.
param.M = 0.
param.sigma = 5.
param.gravity = 1.

nh = param.nh

grid = Grid(param)


f2d = Fluid2d(param, grid)
model = f2d.model


def add_phi(phi, param, grid, x0, y0, sigma, ratio=1, sharpness = 200):
    """ Create a droplet of high density liquid """
    xr, yr = grid.xr, grid.yr
    # ratio controls the ellipticity, ratio=1 is a disc
    r = np.sqrt((xr-param.Lx*x0)**2+(yr-param.Ly*y0)**2*ratio**2)

    y = np.abs(1-np.tanh((r-sigma)*sharpness))
    y /= np.amax(y)
    phi[phi<y] = y[phi<y]

def del_phi(phi, param, grid, x0, y0, sigma, ratio=1, sharpness = 200):
    """ Create a bubble of low density liquid"""
    xr, yr = grid.xr, grid.yr
    # ratio controls the ellipticity, ratio=1 is a disc
    r = np.sqrt((xr-param.Lx*x0)**2+(yr-param.Ly*y0)**2*ratio**2)

    y = np.abs(1+np.tanh((r-sigma)*sharpness))/2
    phi[phi>y] = y[phi>y]



vor = model.var.get('vorticity')
phi = model.var.get('phi')
vor[:, :] = 0.


# =============================================================================
# Distribution of high density liquid 
# =============================================================================

phi[:,:] = 0.

# High density fluid at the bottom:
#phi[:,:]= np.abs(1-np.tanh(((grid.yr)-param.Ly*0.1)*200))/2

# Droplets:
add_phi(phi, param, grid, 0.5, 0.8, 0.05, sharpness=200)

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

