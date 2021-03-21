# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 13:21:06 2021

@author: tugdual
"""
from netCDF4 import Dataset
import matplotlib.pyplot as plt

f = Dataset('/home/tugdual/data/fluid2d/goutte/drop_lines_visco/drop_lines_visco_his.nc')
print(f)
print(f.variables.keys())

phi = f.variables['tracer']
print(phi.dimensions)


plt.figure('goutte', figsize=(8, 8))
plt.clf()
plt.pcolormesh(phi[50, :, :], cmap='inferno', shading='gouraud')