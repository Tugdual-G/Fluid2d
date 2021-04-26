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



def average(x,n):
    X = np.zeros((2*n+1, np.shape(x)[0]+2*n))
    for i in range(n*2):
        X[i+1,0:i+1] = np.nan
        X[0:-i-1, (-i-1)] = np.nan
    for i in range(2*n+1):
        X[i, i:i+len(x)] = x
    print(X)
    return np.nanmean(X,axis=0,keepdims=False)[n:-n]
    
    
#%%
        


enregistrer = False

home = os.environ['HOME']
path = "/data/fluid2d/bien"
print(os.listdir(home + path)) # The name of the dirs are the name of the experiments

tries = 0

while tries < 3:
    try:
        fold = input("Enter the experiment you want to see\n")
        f = Dataset(home + path +'/' + fold + '/' + fold + '_his.nc')
        break
    except:
        print("Incorrect input")
        tries += 1
        if tries > 2:
            exit(0) # If we got too many errors we exit

#%%
#print(f)
#print(f.variables.keys())

phi = f.variables['tracer']
#print(phi)
# Indice de l'image



max_x = np.amax(f.variables['y'])
max_y = np.amax(f.variables['x'])
x = np.linspace(0,max_x,np.shape(phi[1, :, :].T)[1])
y = np.linspace(0,max_y,np.shape(phi[1, :, :].T)[0])
X, Y = np.meshgrid(x,y)
t = np.ravel(f.variables['t'])
i_t = np.nonzero(np.abs(t-0.0)<0.01)[0]
i = i_t[0]
print(i)
y_max = np.load(home + path +'/' + fold + '/' + fold + 'y_max.npy')
y_min = np.load(home + path +'/' + fold + '/' + fold + 'y_min.npy')

plt.figure('goutte3', figsize=(5, 10))
plt.clf()
plt.pcolormesh(X, Y, np.flipud(phi[i, :, :].T), cmap='inferno', shading='gouraud')
plt.colorbar()
#plt.axhline(max_y-y_max[i])
#plt.axhline(max_y-y_min[i])
titre = "t="+str(round(t[i],3))+"s"
plt.title(titre)
plt.tight_layout()
#plt.show()



#plt.savefig('goutte_non-miscible.pdf', dpi=300)
#%%
y_max = np.load(home + path + '/' + fold + '/' + fold + 'y_max.npy')
y_min = np.load(home + path + '/' + fold + '/' + fold + 'y_min.npy')
if enregistrer:
    cmap = 'inferno'
    shade = 'gouraud'
    #Préparation de l'affichage pour enregistrement
    fig1, ax1 = plt.subplots(num = 'animation', dpi = 200, figsize = (4,12))     
    plt.axis('off')
    plt.tight_layout(pad = 0)
    img = ax1.pcolormesh(X,Y,np.flipud(phi[0, :, :].T), cmap = cmap, shading= shade) 
    img1 = ax1.axhline(max_y-y_max[0])
    img2 = ax1.axhline(max_y-y_min[0])
    
    print(int(len(t)/t[-1]))
    
    writer = animation.FFMpegWriter(fps = 5, bitrate = 5000)
    def diapo(n):		
        global img, img1, img2
        img.remove()
        img1.remove()
        img2.remove()             
        img = ax1.pcolormesh(X,Y,np.flipud(phi[n, :, :].T),cmap = cmap, shading = shade) 
        img1 = ax1.axhline(max_y-y_max[n])
        img2 = ax1.axhline(max_y-y_min[n])	
    
    #Création de l'animation
    anim = animation.FuncAnimation(fig1, diapo, frames=50)
    
    file_name = 'longcouloir.mp4'
    anim.save(file_name, writer= writer )

v = [0]
v_x = [0]
v_y = [0]




#%%%

v = np.load(home + path + '/' + fold + '/' + fold + 'velocity.npy')
v_x = np.load(home + path + '/' + fold + '/' + fold + 'velocity_x.npy')
v_y = np.load(home + path + '/' + fold + '/' + fold + 'velocity_y.npy')

plt.figure("Speed", dpi = 200)
"""#First time
ax1 = plt.subplot2grid((4,4), (0,0), colspan=1, rowspan=3)
plt.pcolormesh(X, Y, np.flipud(phi[0, :, :].T), cmap='inferno', shading='gouraud')
ax2 = plt.subplot2grid((4,4), (0,1), colspan=1, rowspan=3, sharey=ax1)
plt.pcolormesh(X, Y, np.flipud(phi[len(t)//3, :, :].T), cmap='inferno', shading='gouraud')
ax3 = plt.subplot2grid((4,4), (0,2), colspan=1, rowspan=3, sharey=ax2)
plt.pcolormesh(X, Y, np.flipud(phi[2*len(t)//3, :, :].T), cmap='inferno', shading='gouraud')
ax4 = plt.subplot2grid((4,4), (0,3), colspan=1, rowspan=3, sharey=ax3, frameon=False)
plt.pcolormesh(X, Y, np.flipud(phi[-1, :, :].T), cmap='inferno', shading='gouraud')
plt.subplot2grid((4,4), (3,0), colspan=4, rowspan=1)"""



#plt.plot(t, v, 'k', linewidth=0.5, label = "Total speed")
plt.plot(t, v_x, 'b', linewidth=0.5, label = "$V_x$")
plt.plot(t, v_y, 'k', linewidth=0.5, label = "$V_y$")
plt.title("Speed of the bubble")
plt.xlabel("t")
plt.ylabel("V")
plt.grid()
plt.tight_layout()
plt.legend()
plt.show()

#%%
phi = f.variables['phi']
nmbr = 5
fig1, ax = plt.subplots(1,nmbr, num = 'snap', clear = True, sharey=True,
                        gridspec_kw = {'hspace': 0.05,'wspace': 0.2 })
i = 0
t_step = len(t)//nmbr
for i in range(nmbr):
    ax[i].pcolormesh(X, Y, np.flipud(phi[t_step*i, :, :].T), cmap='inferno', shading='gouraud')
    ax[i].set_ylim(0.5,3)
    titre = "t="+str(round(t[t_step*i],3))+" s"
    ax[i].set_title(titre)
ax[2].set_xlabel('x')
ax[0].set_ylabel('y')

oscilation = np.load(home + path + '/' + fold + '/' + fold + 'oscilations.npy')
plt.figure("oscillation")
plt.clf()
plt.grid()
#plt.plot(t, oscilation, 'k', linewidth=0.5)
plt.plot(t, average(oscilation,10), 'r', linewidth=0.5)
plt.ylabel("Height")
plt.xlabel("time")
plt.title("Height of the buble in time")
plt.show()

print('dx=',x[1]-x[0])
print(len(oscilation))