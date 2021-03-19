# -*- coding: utf-8 -*-
from param import Param
from grid import Grid
from fluid2d import Fluid2d
import numpy as np
import matplotlib.pyplot as plt
import droplet_operator_simple as dos
import droplet_operator as do

param = Param('default.xml')
param.modelname = 'droplet'
param.expname = 'test'

# domain and resolution
ratio = 1
param.ny = 2**9
param.nx = param.ny*ratio
param.Ly = 1.
param.Lx = param.Ly*ratio
param.npx = 1
param.npy = 1
param.geometry = 'closed'

# time
param.tend = 0.5
param.cfl = 0.8
param.adaptable_dt = True
param.dt = 0.0001
param.dtmax = 0.01

# discretization
param.order = 5
param.timestepping = 'RK3_SSP'

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
param.colorscheme = 'minmax'
param.generate_mp4 = True
param.cmap = 'inferno'

# physics
param.forcing = False
param.noslip = True
param.diffusion = False
param.forcing = False

# rho, sigma and M
param.rho_h = 10.
param.rho_l = 1.
param.M = 0.0
param.sigma = 0.5
param.gravity = 10.
param.nu_h = 10**-6
param.nu_l = 0.

nh = param.nh

grid = Grid(param)





def add_phi(phi, param, grid, x0, y0, sigma, ratio=1, sharpness=200):
    """ Create a droplet of high density liquid """
    xr, yr = grid.xr, grid.yr
    # ratio controls the ellipticity, ratio=1 is a disc
    r = np.sqrt((xr-param.Lx*x0)**2+(yr-param.Ly*y0)**2*ratio**2)

    y = np.abs(1-np.tanh((r-sigma)*sharpness))
    y /= np.amax(y)
    phi[phi < y] = y[phi < y]


def del_phi(phi, param, grid, x0, y0, sigma, ratio=1, sharpness=200):
    """ Create a bubble of low density liquid"""
    xr, yr = grid.xr, grid.yr
    # ratio controls the ellipticity, ratio=1 is a disc
    r = np.sqrt((xr-param.Lx*x0)**2+(yr-param.Ly*y0)**2*ratio**2)

    y = np.abs(1+np.tanh((r-sigma)*sharpness))/2
    phi[phi > y] = y[phi > y]


phi0 = np.zeros((param.ny+6, param.nx+6))


# =============================================================================
# Distribution of high density liquid
# =============================================================================



# High density fluid at the bottom:
#phi[:,:]= np.abs(1-np.tanh(((grid.yr)-param.Ly*0.1)*200))/2

# Droplets:
add_phi(phi0, param, grid, 0.5, 0.7, 0.05, sharpness=100)



def mean_rho(phi, rho_l, rho_h):
    average_phi = np.sum(phi)/(np.shape(phi)[0]*np.shape(phi)[1])
    return average_phi*rho_h + (1-average_phi)*rho_l 
 
param.rho_0 = mean_rho(phi0, param.rho_l, param.rho_h)


f2d = Fluid2d(param, grid)
model = f2d.model

vor = model.var.get('vorticity')
phi = model.var.get('phi')
vor[:, :] = 0.
phi[:, :] = phi0
# =============================================================================
# In order to check the look of the initial conditions.
# =============================================================================


"""dx = param.Ly/param.ny
grad_i, grad_j = dos.gradient_i(phi, dx), dos.gradient_j(phi, dx)
n_i, n_j = do.normalise(grad_i, grad_j)
grad = np.sqrt(grad_i**2 + grad_j**2)"""

# computing the torque:
#trq = np.zeros_like(phi)
#torque(trq, phi, dx, rho_l = 1, rho_h = 10, xi = 5*dx, sigma = 0.01)

"""plt.figure(figsize=(8, 8))"""

# Check phi:
"""plt.pcolormesh(grid.xr, grid.yr, grad**5, shading='nearest')"""
# plt.colorbar()

# Check the torque:
#plt.pcolormesh(grid.xr, grid.yr, trq, shading = 'nearest')

# Check the source term in equation (1d) :
#plt.pcolormesh(grid.xr, grid.yr, source_1d(phi, dx, xi = 3*dx, M = 0.01), shading = 'nearest')


# Check the gradient:
"""grad_i[grad < 40] = 0
grad_j[grad < 40] = 0"""
#plt.quiver(grid.xr, grid.yr, -grad_j, -grad_i, scale=500, minlength=0)


model.set_psi_from_vorticity()

f2d.loop()

# Check phi:
#plt.figure(figsize=(12, 10))
#plt.pcolormesh(grid.xr, grid.yr, phi, cmap='inferno', shading='gouraud')
# plt.colorbar()
