# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 13:21:06 2021

@author: tugdual
"""
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt


import os


def get_center(phi_at_t, treshold):
    """ gets the barycenter of phi, over a chosen threshold """
    sum_phix = 0
    sum_phiy = 0
    sum_phi = 0
    for i in range(len(phi_at_t[:, 0])):
        for j in range(len(phi_at_t[0, :])):
            if phi_at_t[i, j] < treshold:
                # i and j are in the wrong order because of how we obtain phi
                sum_phiy += y[i] * phi_at_t[i, j]
                sum_phix += x[j] * phi_at_t[i, j]
                sum_phi += phi_at_t[i, j]

    return sum_phix / sum_phi, sum_phiy / sum_phi


def get_min_max_phi(phi_at_t, threshold, index_x):
    """ outputs the y minimum and maximum of phi if its above a given treshold
    """
    y_min = 100
    y_max = 0
    max_done = False
    iterate = True
    j = len(phi_at_t[:,  index_x])-1

    while iterate and j > 0:
        # We iterate over j, until we have found the minimum and the maximum
        # We should always find a value because we take the index of the center
        # but the j > 0 avoids any crash
        if max_done is False and phi_at_t[j, index_x] > threshold:
            y_max = y[j]
            max_done = True

        # f max_done:
            # rint(phi_at_t[j, i])

        if max_done and phi_at_t[j, index_x] < threshold:

            y_min = y[j+1]
            iterate = False
        j -= 1

    return y_min, y_max


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

i = 0

max_x = np.amax(f.variables['y'])
max_y = np.amax(f.variables['x'])

# print(max_x, max_y)

x = np.linspace(0, max_x, np.shape(phi[1, :, :].T)[1])
y = np.linspace(0, max_y, np.shape(phi[1, :, :].T)[0])

dx = max_x / len(x)
dy = max_y / len(y)

X, Y = np.meshgrid(x, y)
t = np.ravel(f.variables['t'])

x_center, y_center = get_center(phi[i, :, :].T, 0.8)

indices_x = np.argwhere((x > x_center - dx) * (x < x_center + dx))
indices_y = np.argwhere((y > y_center - dy) * (y < y_center + dy))

y_min, y_max = get_min_max_phi(phi[i, :, :].T, 0.8, indices_x[0][0])
hauteur = y_max - y_min
x_min, x_max = get_min_max_phi(phi[i, :, :], 0.8, indices_y[0][0])
largeur = x_max - x_min

plt.figure('goutte', figsize=(5, 10))
plt.clf()
plt.pcolormesh(X, Y, np.flipud(phi[i, :, :].T), cmap='inferno', shading='gouraud')
plt.colorbar()
plt.plot(x_center, y[-1]-y_center, "ro")
# print(max_y, y_max, y_min)
plt.axhline(max_y-y_max)
plt.axhline(max_y-y_min)
# print(x_center, y_center)
titre = "t="+str(round(t[i], 3))+"s"
plt.title(titre)
plt.tight_layout()

plt.show()

#%%

v = [0]
v_x = [0]
v_y = [0]

prev_x, prev_y = get_center(phi[0, :, :].T, 0.8)

if hauteur > largeur:
    a = hauteur
    b = largeur
else:
    a = largeur
    b = hauteur

ellipticity = [1 - b/a]
list_y_max = []
list_y_min = []

# Computation of different speed
for i in range(1, len(t)):
    print(i, "/", len(t))  # Can be used to show progress
    x_i, y_i = get_center(phi[i, :, :].T, 0.8)
    # .T should be removed in case of a change in the direction of g in the experiment
    v_x.append((-x_i + prev_x) / (t[i] - t[i-1]))
    # get speed by the difference of position / difference in time
    v_y.append((-y_i + prev_y) / (t[i] - t[i-1]))
    v.append((np.sqrt((-x_i + prev_x)**2 + (-y_i + prev_y)**2)) / (t[i] - t[i-1]))
    prev_x = x_i
    prev_y = y_i

    indices_x = np.argwhere((x > x_i - dx) * (x < x_i + dx))
    indices_y = np.argwhere((y > y_i - dy) * (y < y_i + dy))

    y_min, y_max = get_min_max_phi(phi[i, :, :].T, 0.8, indices_x[0][0])
    hauteur = y_max - y_min
    x_min, x_max = get_min_max_phi(phi[i, :, :], 0.8, indices_y[0][0])
    largeur = x_max - x_min

    if hauteur > largeur:
        a = hauteur
        b = largeur
    else:
        a = largeur
        b = hauteur

    ellipticity.append(1 - b/a)
    list_y_max += [y_max]
    list_y_min += [y_min]

np.save(home + path + '/' + fold + '/' + fold + 'velocity.npy', v)
np.save(home + path + '/' + fold + '/' + fold + 'velocity_x.npy', v_x)
np.save(home + path + '/' + fold + '/' + fold + 'velocity_y.npy', v_y)

np.save(home + path + '/' + fold + '/' + fold + 'ellipticity.npy', ellipticity)
np.save(home + path + '/' + fold + '/' + fold + 'y_max.npy', list_y_max)
np.save(home + path + '/' + fold + '/' + fold + 'y_min.npy', list_y_min)


#%%
# oscilation = []
# list_y_max = []
# list_y_min = []

# for i in range(len(t)):
    # print(i, "/", len(t)) # Can be used to show progress
    # y_min_i, y_max_i = get_min_max_phi(phi[i, :, :].T, 0.95) #.T should be removed in case of a change in the direction of g in the experiment
    # # print(y_max_i, y_min_i, y_max_i - y_min_i)
    # oscilation.append(y_max_i - y_min_i)
    # list_y_max += [y_max]
    # list_y_min += [y_min]

# np.save(home + path + '/' + fold + '/' + fold + 'oscilations.npy', oscilation)
# np.save(home + path + '/' + fold + '/' + fold + 'y_max.npy', list_y_max)
# np.save(home + path + '/' + fold + '/' + fold + 'y_min.npy', list_y_min)
