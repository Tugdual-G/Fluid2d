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
ratio = 1
param.ny = 2**8
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
param.var_to_save = ['vorticity', 'psi', 'u', 'v', 'phi']
param.list_diag = 'all'
param.freq_plot = 5
param.freq_his = .02
param.freq_diag = .1

# plot
param.plot_interactive = True
param.plot_var = 'phi'
param.cax = [-0.5, 1.5]
param.colorscheme = 'imposed'
param.generate_mp4 = False
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
phi = model.var.get('phi')

# control parameters of the experiment
N = 4.  # Brunt Vaisala frequency squared
S = 20.  # vertical shear


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

# Creating a disc of buoyancy: 
buoy = np.zeros_like(phi)
dtype = 'smooth_step'
sigma = 0.2*param.Lx

buoy += -1*set_buoyancy(param, grid, 0.5, 0.5, sigma,
                dtype, ratio=1, sharpness = 100)


def get_phi(buoy):
    phi =  np.amax(buoy)-buoy
    return phi/(np.amax(phi)-np.amin(phi))

phi[:,:] = get_phi(buoy)



print('Ri = %4.2f' % (N**2/S**2))

vor[:, :] = 0.



dx = param.Ly/param.ny
grad_i, grad_j = gradient(phi, dx)
n_i, n_j = normalise( grad_i, grad_j)
grad = np.sqrt(grad_i**2 + grad_j**2)

#on récupère le torque
trq = np.zeros_like(phi)
torque(trq, phi, dx, rho_l = 1, rho_h = 10, xi = 5*dx, sigma = 0.01)

# =============================================================================
# In order to check the look of the initial conditions.
# =============================================================================


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