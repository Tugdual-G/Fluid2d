# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 13:21:06 2021

@author: tugdual
"""
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif') 

enregistrer = False
f = Dataset('/home/tugdual/data/fluid2d/bien/smooth_startbiencool/smooth_start_q8-tracerlarge_lapladiago_ordr3_his.nc')
#print(f)
#print(f.variables.keys())

phi = f.variables['phi']
#print(phi.dimensions)
# Indice de l'image
i = 350

x = np.linspace(0,1,np.shape(phi[1, :, :].T)[1])
y = np.linspace(0,2,np.shape(phi[1, :, :].T)[0])
X, Y = np.meshgrid(x,y)
t = np.ravel(f.variables['t'])


plt.figure('goutte', figsize=(5, 10))
plt.clf()
ax = plt.subplot(111)

ax.pcolormesh(X, Y, np.flipud(phi[i, :, :].T), cmap='inferno', shading='gouraud')
titre = "t="+str(round(t[i],3))+"s"
ax.set_title(titre)
plt.tight_layout()
#ax.set_aspect(1)
plt.savefig('test.png')


if enregistrer:
    cmap = 'inferno'
    shade = 'gouraud'
    #Préparation de l'affichage pour enregistrement
    fig1, ax1 = plt.subplots(num = 'animation',figsize = (5,10))     
    plt.axis('off')
    plt.tight_layout(pad = 0)
    img = ax1.pcolormesh(X,Y,np.flipud(phi[0, :, :].T), cmap = cmap, shading= shade) 
    
    
    print(int(len(t)/t[-1]))
    
    writer = animation.FFMpegWriter(fps = 50, bitrate = 500)
    def diapo(n):		
        global img
        img.remove()
        img = ax1.pcolormesh(X,Y,np.flipud(phi[n, :, :].T),cmap = cmap, shading = shade) 
    		
    #Création de l'animation
    anim = animation.FuncAnimation(fig1, diapo, frames= len(t), repeat = False)
    
    file_name = 'miscible.mp4'
    anim.save(file_name, writer= writer )