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

def get_center(phi_at_t, treshold):
    """ gets the barycenter of phi, over a chosen threshold """
    sum_phix = 0
    sum_phiy = 0
    sum_phi = 0
    for i in range(len(phi_at_t[:,0])):
        for j in range(len(phi_at_t[0,:])):
            if phi_at_t[i,j] > treshold:
                # i and j are in the wrong order because of how we obtain phi
                sum_phiy += max_x * j / len(x)  * phi_at_t[i,j]
                # we normalize j or i an integer, to the real distance between two points
                sum_phix += max_y * i / len(y)  * phi_at_t[i,j]
                sum_phi += phi_at_t[i,j]

    return sum_phix/ sum_phi, sum_phiy/ sum_phi

def get_min_max_phi(phi_at_t, threshold):
    """ outputs the y minimum and maximum of phi if its above a given treshold """
    y_min = 100
    y_max = 0

    for i in range(len(phi_at_t[0,:])):
        # We iterate over all possible x to find the max of y and its min
        max_done = False
        iterate = True
        j = len(phi_at_t[:,0]) - 1
        while iterate and j > 0:
            # We iterate over j, until we have found the minimum and the maximum (or if we found nothing above the threshold
            if max_done == False and phi_at_t[j, i] > threshold:

                if y_max < max_y * j / len(y):
                    y_max = max_y * j / len(y)
                max_done = True

            #if max_done:
                #print(phi_at_t[j, i])

            if max_done and phi_at_t[j, i] < threshold:

                if y_min > max_y * (j+1) / len(y):
                    y_min = max_y * (j+1) / len(y)
                iterate = False
            j -= 1

    return y_min, y_max

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
max_y = 2

x = np.linspace(0,max_x,np.shape(phi[1, :, :].T)[1])
y = np.linspace(0,max_y,np.shape(phi[1, :, :].T)[0])
X, Y = np.meshgrid(x,y)
t = np.ravel(f.variables['t'])

x_center, y_center = get_center(phi[i, :, :].T, 0.8)

plt.figure('goutte', figsize=(5, 10))
plt.clf()
plt.pcolormesh(X, Y, np.flipud(phi[i, :, :].T), cmap='inferno', shading='gouraud')
plt.colorbar()
#plt.plot(y_center, 2 - x_center, "ro")
titre = "t="+str(round(t[i],3))+"s"
plt.title(titre)
plt.tight_layout()

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

prev_x, prev_y = get_center(phi[0, :, :], 0.8)

#Computation of different speed
for i in range(1,len(t)):
    print(i, "/", len(t)) # Can be used to show progress
    x_i, y_i = get_center(phi[i, :, :], 0.8)
    v_x.append((x_i - prev_x) / (t[i] - t[i-1]))
    v_y.append((y_i - prev_y) / (t[i] - t[i-1]))
    v.append((np.sqrt((x_i - prev_x)**2 + (y_i - prev_y)**2)) / (t[i] - t[i-1]))
    prev_x = x_i
    prev_y = y_i

plt.figure("Speed")
plt.plot(t, v, label = "Total speed")
plt.plot(t, v_x, label = "Vx")
plt.plot(t, v_y, label = "Vy")
plt.title("Speed of the bubble")
plt.xlabel("Time")
plt.ylabel("Speed")
plt.legend()
plt.show()

oscilation = []

for i in range(len(t)):
    print(i, "/", len(t)) # Can be used to show progress
    y_min_i, y_max_i = get_min_max_phi(phi[i, :, :].T, 0.97)
    print(y_max_i, y_min_i, y_max_i - y_min_i)
    oscilation.append(y_max_i - y_min_i)

plt.figure("oscillation")
plt.plot(t, oscilation)
plt.ylabel("Height")
plt.xlabel("time")
plt.title("Height of the buble in time")
plt.show()
