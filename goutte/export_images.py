# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 13:21:06 2021

@author: tugdual
"""
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

enregistrer = False

f = Dataset('/home/tugdual/data/fluid2d/smooth_start_q8-tracerlarge_lapladiago/smooth_start_q8-tracerlarge_lapladiago_his.nc')
#print(f)
#print(f.variables.keys())

phi = f.variables['tracer']
#print(phi.dimensions)
# Indice de l'image
i = 200

x = np.linspace(0,1,np.shape(phi[1, :, :].T)[1])
y = np.linspace(0,2,np.shape(phi[1, :, :].T)[0])
X, Y = np.meshgrid(x,y)
t = np.ravel(f.variables['t'])


plt.figure('goutte', figsize=(5, 10))
plt.clf()
plt.pcolormesh(X, Y, np.flipud(phi[i, :, :].T), cmap='inferno', shading='gouraud')
titre = "t="+str(round(t[i],3))+"s"
plt.title(titre)
plt.tight_layout()



if enregistrer:
    cmap = 'inferno'
    shade = 'gouraud'
    #Préparation de l'affichage pour enregistrement
    fig1, ax1 = plt.subplots(num = 'animation', dpi = 200, figsize = (8,16))     
    plt.axis('off')
    plt.tight_layout(pad = 0)
    img = ax1.pcolormesh(X,Y,np.flipud(phi[0, :, :].T), cmap = cmap, shading= shade) 
    
    print(int(len(t)/t[-1]))
    
    writer = animation.FFMpegWriter(fps = 50, bitrate = 5000)
    def diapo(n):		
        global img
        img.remove()
        img = ax1.pcolormesh(X,Y,np.flipud(phi[n, :, :].T),cmap = cmap, shading = shade) 
    		
    #Création de l'animation
    anim = animation.FuncAnimation(fig1, diapo, frames= len(t), repeat = True)
    
    file_name = 'gtte.mp4'
    anim.save(file_name, writer= writer )