# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 13:21:06 2021

@author: tugdual
"""
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

enregistrer = False

home = os.environ['HOME']
print(os.listdir(home + "/data/fluid2d")) # The name of the dirs are the name of the experiments

tries = 0

while tries < 3:
    try:
        fold = input("Enter the experiment you want to see\n")
        f = Dataset(home + '/data/fluid2d/' + fold + '/' + fold + '_his.nc')
        break
    except:
        print("Incorrect input")
        tries += 1
        if tries > 2:
            exit(0) # If we got too many errors we exit


#print(f)
#print(f.variables.keys())

phi = f.variables['tracer']
#print(phi)
# Indice de l'image
i = 200

max_x = 1
max_y = 3

x = np.linspace(0,max_x,np.shape(phi[1, :, :].T)[1])
y = np.linspace(0,max_y,np.shape(phi[1, :, :].T)[0])
X, Y = np.meshgrid(x,y)
t = np.ravel(f.variables['t'])


#plt.figure('goutte', figsize=(5, 10))
#plt.clf()
#plt.pcolormesh(X, Y, np.flipud(phi[i, :, :].T), cmap='inferno', shading='gouraud')
#plt.colorbar()
#titre = "t="+str(round(t[i],3))+"s"
#plt.title(titre)
#plt.tight_layout()
#plt.show()




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

v = [0]
v_x = [0]
v_y = [0]




#%%%

v = np.load(home + '/data/fluid2d/' + fold + '/' + fold + 'velocity.npy')
v_x = np.load(home + '/data/fluid2d/' + fold + '/' + fold + 'velocity_x.npy')
v_y = np.load(home + '/data/fluid2d/' + fold + '/' + fold + 'velocity_y.npy')

plt.figure("Speed", dpi = 200)
#First time
ax1 = plt.subplot2grid((4,4), (0,0), colspan=1, rowspan=3)
plt.pcolormesh(X, Y, np.flipud(phi[0, :, :].T), cmap='inferno', shading='gouraud')
ax2 = plt.subplot2grid((4,4), (0,1), colspan=1, rowspan=3, sharey=ax1)
plt.pcolormesh(X, Y, np.flipud(phi[len(t)//3, :, :].T), cmap='inferno', shading='gouraud')
ax3 = plt.subplot2grid((4,4), (0,2), colspan=1, rowspan=3, sharey=ax2)
plt.pcolormesh(X, Y, np.flipud(phi[2*len(t)//3, :, :].T), cmap='inferno', shading='gouraud')
ax4 = plt.subplot2grid((4,4), (0,3), colspan=1, rowspan=3, sharey=ax3, frameon=False)
plt.pcolormesh(X, Y, np.flipud(phi[-1, :, :].T), cmap='inferno', shading='gouraud')

plt.subplot2grid((4,4), (3,0), colspan=4, rowspan=1)

#plt.plot(t, v, 'k', linewidth=0.5, label = "Total speed")
plt.plot(t, v_x, 'b', linewidth=0.5, label = "$V_x$")
plt.plot(t, v_y, 'k', linewidth=0.5, label = "$V_y$")
plt.title("Speed of the bubble")
plt.xlabel("Time")
plt.ylabel("Speed")
plt.legend()
plt.show()

#%%



oscilation = np.load(home + '/data/fluid2d/' + fold + '/' + fold + 'oscilations.npy')
plt.figure("oscillation")
plt.clf()
plt.plot(t, oscilation, 'k', linewidth=0.5)
plt.ylabel("Height")
plt.xlabel("time")
plt.title("Height of the buble in time")
plt.show()
