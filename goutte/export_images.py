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
plt.rc('text', usetex=True)
plt.rc('font', family='serif')



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
print(os.listdir(home + path))# The name of the dirs are the name of the experiments

# fold = ""
# f = Dataset(home + path +'/' + fold + '/' + fold + '_his.nc')

tries = 0

while tries < 3:
    try:
        fold = input("Enter the experiment you want to see\n")
        f = Dataset(home + path + '/' + fold + '/' + fold + '_his.nc')
        break
    except:
        print("Incorrect input")
        tries += 1
        if tries > 2:
            exit(0)  # If we got too many errors we exit

#%%
# print(f)
# print(f.variables.keys())

phi = f.variables['phi']
# print(phi)


max_x = np.amax(f.variables['y'])
max_y = np.amax(f.variables['x'])
x = np.linspace(0, max_x, np.shape(phi[1, :, :].T)[1])
y = np.linspace(0, max_y, np.shape(phi[1, :, :].T)[0])
X, Y = np.meshgrid(x, y)
t = np.ravel(f.variables['t'])
i_t = np.nonzero(np.abs(t-0.6)<0.01)[0]
i = i_t[0]
print(i)


plt.figure('goutte')

plt.clf()
plt.pcolormesh(X, Y, np.flipud(phi[i, :, :].T), cmap='inferno', shading='gouraud')
plt.colorbar(label='Densité de tracer')
plt.xlabel('x')
#plt.ylabel('y')
# plt.axhline(max_y-y_max[i])
# plt.axhline(max_y-y_min[i])
axes = plt.gca()
axes.set_aspect('equal', 'box')
plt.yticks([0,0.25,0.5,0.75,1,1.25,1.5,1.75],['','','','','','','',''])
titre = "t="+str(round(t[i], 3))+"s"
plt.title(titre)
plt.tight_layout(pad=1)
# plt.show()

plt.savefig(home + path + '/' + fold + '/' + 'view.pdf', dpi=300)

#%%
y_max = np.load(home + path + '/' + fold + '/' + fold + 'y_max.npy')
y_min = np.load(home + path + '/' + fold + '/' + fold + 'y_min.npy')
if enregistrer:
    cmap = 'inferno'
    shade = 'gouraud'
    # Préparation de l'affichage pour enregistrement
    fig1, ax1 = plt.subplots(num='animation', dpi=200, figsize=(4, 12))
    plt.axis('off')
    plt.tight_layout(pad=0)
    img = ax1.pcolormesh(X, Y, np.flipud(phi[0, :, :].T), cmap=cmap, shading=shade)
    img1 = ax1.axhline(max_y-y_max[0])
    img2 = ax1.axhline(max_y-y_min[0])

    print(int(len(t)/t[-1]))

    writer = animation.FFMpegWriter(fps=5, bitrate=5000)

    def diapo(n):
        global img, img1, img2
        img.remove()
        img1.remove()
        img2.remove()            
        img = ax1.pcolormesh(X, Y, np.flipud(phi[n, :, :].T), cmap=cmap, shading=shade)
        img1 = ax1.axhline(max_y-y_max[n])
        img2 = ax1.axhline(max_y-y_min[n])

    # Création de l'animation
    anim = animation.FuncAnimation(fig1, diapo, frames=50)

    file_name = fold + '.mp4'
    anim.save(file_name, writer=writer)

v = [0]
v_x = [0]
v_y = [0]




#%%%

v_x = np.load(home + path + '/' + fold + '/' + fold + 'velocity_x.npy')
v_y = np.load(home + path + '/' + fold + '/' + fold + 'velocity_y.npy')
v_x = average(v_x,40)
v_y = average(v_y,40)


oscillation = np.load(home + path + '/' + fold + '/' + fold + 'oscillations.npy')
oscillation = average(oscillation,10)
fig, ax = plt.subplots(2,1, num='speed', sharex = True, clear=True)

ax[0].plot(t, v_x, 'b', linewidth=0.5, label = "$V_x$")
ax[0].plot(t, v_y, 'k', linewidth=0.5, label = "$V_y$")
ax[0].set_title("Vitesse de la bulle", loc='left')
ax[0].set_ylabel("V")
#ax[0].set_xlabel("t")
ax[0].grid()
ax[0].legend()
# plt.show()

# plt.figure("oscillation")
# plt.clf()
ax[1].plot(t, 1-oscillation, 'k', linewidth=0.5)
ax[1].grid()
ax[1].set_ylabel("1-H/L")
ax[1].set_xlabel("t")
ax[1].set_title("Ellipticité", loc='left', pad=5)

plt.show()
fig.savefig(home + path + '/' + fold + '/' + 'oscill.pdf', dpi = 300)
#%%
phi = f.variables['phi']
nmbr = 5
fig1, ax = plt.subplots(1, nmbr, num='snap', clear=True, sharey=True,
                        gridspec_kw={'hspace': 0.05, 'wspace': 0.2})
i = 0
t_step = len(t)//nmbr
for i in range(nmbr):
    pcm = ax[i].pcolormesh(X, Y, np.flipud(phi[t_step*i, :, :].T), cmap='inferno', shading='gouraud')
    titre = "t="+str(round(t[t_step*i], 3))+" s"
    ax[i].set_title(titre)
    ax[i].axis('equal')
ax[2].set_xlabel('x')
ax[0].set_ylabel('y')
#fig.colorbar(pcm, ax=ax[-1], label='Densité de tracer', )
fig1.savefig(home + path + '/' + fold + '/' + 'snapshot.pdf', dpi = 300)