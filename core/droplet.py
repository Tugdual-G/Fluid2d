from operators import Operators
from variables import Var
from timescheme import Timescheme
from importlib import import_module
import fortran_diag as fd
import fortran_advection as fa
import numpy as np


class Droplet(object):
    """ Model for a two-phases liquid.

    It provides the step(t,dt) function
    and 'var' containing the mode state
    """

    def __init__(self, param, grid):

        self.list_param = ['forcing', 'noslip', 'timestepping',
                           'alphaT', 'betaS',
                           'diffusion', 'Kdiff', 'myrank',
                           'forcing_module', 'gravity', 'isisland',
                           'customized', 'custom_module', 'additional_tracer',
                           'rho_h', 'rho_l', 'rho_0', 'M', 'sigma', 'nu_h', 'nu_l', 'n_xi']
        param.copy(self, self.list_param)

        # for potential energy
        self.list_param = ['xr', 'yr', 'nh', 'Lx', 'msk', 'area', 'mpitools']
        grid.copy(self, self.list_param)

        # for variables
        param.varname_list = ['vorticity',
                              'psi', 'u', 'v', 'density', 'phi', 'tracer']
        param.tracer_list = ['vorticity', 'phi', 'tracer']
        param.whosetspsi = ('vorticity')

        if hasattr(self, 'additional_tracer'):
            for k in range(len(self.additional_tracer)):
                trac = self.additional_tracer[k]
                param.varname_list.append(trac)
                param.tracer_list.append(trac)

        self.varname_list = param.varname_list

        param.sizevar = [grid.nyl, grid.nxl]
        self.var = Var(param)
        dref = self.var.get('density').copy()
        self.dref = dref
        self.source = np.zeros(param.sizevar)

        # for operators
        self.ope = Operators(param, grid)
        self.dx = param.Lx/param.nx
        # for timescheme
        self.tscheme = Timescheme(param, self.var.state)
        self.tscheme.set(self.dynamics, self.timestepping)

        if self.forcing:
            if self.forcing_module == 'embedded':
                print(
                    'Warning: check that you have indeed added the forcing to the model')
                print('Right below the line    : model = f2d.model')
                print('you should have the line: model.forc = Forcing(param, grid)')

                pass
            else:
                try:
                    f = import_module(self.forcing_module)

                except ImportError:
                    print('module %s for forcing cannot be found'
                          % self.forcing_module)
                    print('make sure file **%s.py** exists' %
                          self.forcing_module)
                    exit(0)

                self.forc = f.Forcing(param, grid)

        self.diags = {}

        if self.customized:
            try:
                f = import_module(self.custom_module)
                print(f)
                self.extrastep = f.Step(param, grid)
            except ImportError:
                print('module %s for forcing cannot be found'
                      % self.custom_module)
                print('make sure file **%s.py** exists' % self.custom_module)
                exit(0)

    def step(self, t, dt):

        # 1/ integrate advection
        self.tscheme.forward(self.var.state, t, dt)

        # 2/ integrate source
        if self.noslip:
            self.add_noslip(self.var.state)

        if self.customized:
            self.extrastep.do(self.var, t, dt)

    def dynamics(self, x, t, dxdt):
        """ Gives the right hand side to integrate in dxdt"""

        self.ope.rhs_adv(x, t, dxdt)

        # db/dx is a source term for the vorticity

        self.ope.rhs_droplet(x, t, dxdt, xi=self.n_xi*self.dx,
                             sigma=self.sigma, M=self.M, rho_l=self.rho_l,
                             rho_h=self.rho_h, rho_0=self.rho_0, g=- self.gravity,
                             nu_h=self.nu_h, nu_l=self.nu_l)

        self.ope.invert_vorticity(dxdt, flag='fast')

    def add_noslip(self, x):
        self.ope.rhs_noslip(x, self.source)
        self.ope.invert_vorticity(x, flag='fast', island=self.isisland)

    def set_psi_from_vorticity(self):
        self.ope.invert_vorticity(self.var.state, island=self.isisland)

    def diagnostics(self, var, t):
        """ should provide at least 'maxspeed' (for cfl determination) """

        nh = self.nh
        u = var.get('u')
        v = var.get('v')
        vort = var.get('vorticity')
        dens = var.get('density')

        ke, maxu = fd.computekemaxu(self.msk, u, v, self.nh)

        z, z2 = fd.computesumandnorm(self.msk, vort, self.nh)

        b, b2 = fd.computesumandnorm(self.msk, dens, self.nh)

        #  potential energy
        pe = + self.gravity * fd.computesum(self.msk, dens*self.yr, nh)

        cst = self.mpitools.local_to_global([(maxu, 'max'), (ke, 'sum'),
                                             (z, 'sum'), (z2, 'sum'),
                                             (pe, 'sum'), (b, 'sum'),
                                             (b2, 'sum')])

        self.diags['maxspeed'] = cst[0]
        self.diags['ke'] = cst[1] / self.area
        self.diags['pe'] = cst[4] / self.area
        self.diags['energy'] = (cst[1]+cst[4]) / self.area
        self.diags['vorticity'] = cst[2] / self.area
        self.diags['enstrophy'] = 0.5*cst[3] / self.area
        self.diags['density'] = cst[5] / self.area
        self.diags['drms'] = np.sqrt(cst[6] / self.area-(cst[5]/self.area)**2)
