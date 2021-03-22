# -*- coding: utf-8 -*-
from param import Param
from grid import Grid
from fluid2d import Fluid2d
import numpy as np
import matplotlib.pyplot as plt
import droplet_operator_lapla_diago as dold
import droplet_operator as do
from random import uniform 


param = Param('default.xml')
param.modelname = 'droplet'
#name = str(input('nom export:'))
param.expname = 'rho1-2_sig008_q9'

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
param.tend = 1.
param.cfl = 1.5
param.adaptable_dt = True
param.dt = 0.0001
param.dtmax = 0.01

# discretization
param.order = 5
param.timestepping = 'RK3_SSP'

# output
param.var_to_save = ["phi"]
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
param.cmap = 'inferno'

# physics
param.forcing = False
param.noslip = True
param.diffusion = False
param.forcing = False

# rho, sigma and M
param.rho_h = 1.2
param.rho_l = 1.
param.M = 0.0
param.sigma = 0.08
param.gravity = 10.
param.nu_h = 0.
param.nu_l = 0.
param.n_xi = 3.

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
#phi0[:,:] = np.abs(1-np.tanh(((grid.yr)-param.Ly*0.5)*200))/2
#phi0[:,:] += -np.abs(1-np.tanh(((grid.yr)-param.Ly*0.4)*200))/2

# Droplets:
add_phi(phi0, param, grid, 0.5, 0.7, 0.05, sharpness=100)


def mean_rho(phi, rho_l, rho_h):
    return np.mean(phi)*rho_h + (1-np.mean(phi))*rho_l 
 
param.rho_0 = mean_rho(phi0, param.rho_l, param.rho_h)


f2d = Fluid2d(param, grid)
model = f2d.model

tracer = model.var.get('tracer')
vor = model.var.get('vorticity')
phi = model.var.get('phi')
vor[:, :] = 0.
phi[:, :] = phi0

x = np.arange(0, np.shape(tracer)[0], 1)
X , Y = np.meshgrid(x,x)
 

# =============================================================================
#  Set tracer
# =============================================================================

tracer[:,:] = 0.5*np.cos((grid.xr-0.5)*200)**3*np.cos((grid.xr-0.5)*3)**4

#add_phi(tracer, param, grid, 0.5, 0.85, 0.06, sharpness=100)

# =============================================================================
# In order to check the look of the initial conditions.
# =============================================================================


"""dx = param.Ly/param.ny

F_i, F_j = do.surface_tension(phi, dx, param.n_xi, param.sigma)


fig , ax = plt.subplots(1,2, figsize=(20, 10),num ='test', clear = True)


ax[0].pcolormesh(grid.xr, grid.yr, F_i**6+F_j**6, shading='nearest')


F_i, F_j = dold.surface_tension(phi, dx, param.n_xi, param.sigma)


ax[1].pcolormesh(grid.xr, grid.yr, F_i**6+F_j**6, shading='nearest')


#ax.quiver(grid.xr, grid.yr, F_j, F_i, minlength=0)"""


# =============================================================================
# Start
# =============================================================================

model.set_psi_from_vorticity()

f2d.loop()


