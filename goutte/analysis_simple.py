# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 13:21:06 2021

@author: tugdual
"""
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt


import os


def get_center(phi, treshold, Lx, Ly):
    """ gets the barycenter of phi, over a chosen threshold """
    sum_phix = 0
    sum_phiy = 0
    sum_phi = 0
    for i in range(len(phi[:, 0])):
        for j in range(len(phi[0, :])):
            if phi[i, j] > treshold:
                # i and j are in the wrong order because of how we obtain phi
                sum_phiy += i * phi[i, j]
                sum_phix += j * phi[i, j]
                sum_phi += phi[i, j]
    
    xc = (sum_phix / sum_phi)*(Lx/len(phi[0,:]))
    yc = (sum_phiy / sum_phi)*(Ly/len(phi[:,0]))
    jc, ic = round(sum_phix / sum_phi), round(sum_phiy / sum_phi)
    
    return jc, ic, xc, yc


def get_W_phi(phi, threshold, index_x, index_y):
    """ outputs the y minimum and maximum of phi if its above a given treshold
    """

    j_sup = index_x
    j_inf = index_x
    
    i = 1
    # width:
    while j_inf > 1 and i>0 :
        mean_phi = phi[index_y,j_inf] + phi[index_y,j_inf-1] + \
            phi[index_y,j_inf+1] + phi[index_y-1,j_inf] + phi[index_y+1,j_inf]
        mean_phi = mean_phi/5
                
        if mean_phi < threshold:
            i = -1          
        else:
            j_inf -= 1

    i = 1
    while j_sup+1 < np.shape(phi)[1] and i>0 :
        mean_phi = phi[index_y,j_sup] + phi[index_y,j_sup-1] + \
            phi[index_y,j_sup+1] + phi[index_y-1,j_sup] + phi[index_y+1,j_sup]
        mean_phi = mean_phi/5
                
        if mean_phi < threshold:

            i = -1
        else:
            j_sup += 1        
                
    return int(j_inf), int(j_sup)


#%%

home = os.environ['HOME']

path = "/data/fluid2d/bien"
# path = "/data/fluid2d"
print(os.listdir(home + path))  # The name of the dirs are the name of the experiments

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

# print(f)
# print(f.variables.keys())

phi = f.variables['phi']
# print(phi)
# Indice de l'image

#%%

i = 9



x = f.variables['y']
x = x[:]
y = f.variables['x']
y = y[:]

t = np.ravel(f.variables['t'])
X, Y = np.meshgrid(x, y)

ic,jc, xc, yc = get_center(phi[i,:,:].T,0.8,np.amax(x), np.amax(y))


j_inf, j_sup = get_W_phi(phi[i,:,:].T,0.5, ic,jc)
i_inf, i_sup = get_W_phi(phi[i,:,:],0.5,jc, ic)

plt.pcolormesh(X, Y, np.flipud(phi[i, :, :].T), cmap='inferno', shading='nearest')
plt.axis('equal')
plt.plot(xc,np.amax(y)-yc,'xr')
plt.axvline(x[j_inf])
plt.axvline(x[j_sup])
plt.axhline(np.amax(y)-y[i_inf])
plt.axhline(np.amax(y)-y[i_sup])
print(1 - abs((i_sup-i_inf)/(j_sup-j_inf)))

#%%

v = [0]

prev_x, prev_y = get_center(phi[0, :, :].T, 0.8)


ellipticity = np.zeros_like(t[:])

for i in range(len(t)):
    xc, yc = get_center(phi[i,:,:].T,0.8)
    j_inf, j_sup = get_W_phi(phi[i,:,:].T,0.5, xc,yc)
    i_inf, i_sup = get_W_phi(phi[i,:,:],0.5,yc, xc)
    ellipticity[i] = 1 - abs((i_sup-i_inf)/(j_sup-j_inf))
    print(i,'/',len(t))
    print(ellipticity[i])

np.save(home + path + '/' + fold + '/' + fold + 'ellipticity.npy', ellipticity)

#%%
print(ellipticity)
plt.plot(t[:450],ellipticity[:450])