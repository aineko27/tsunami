# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 14:08:34 2016

@author: seis
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from function_tunami import *
import time
import sys
fig = plt.figure()
time1 = time.time()

#時間刻み幅、空間刻み幅などを決める
g = 9.80665
x_num = 40
y_num = 40
dt = 0.01
dx = 1
dy = 1
steps = 100
D = np.ones([x_num, y_num])*-200
h_f = np.zeros([x_num, y_num])
M_f = np.zeros([x_num, y_num])
N_f = np.zeros([x_num, y_num])
h_a = np.zeros([x_num, y_num])
M_a = np.zeros([x_num, y_num])
N_a = np.zeros([x_num, y_num])
RMSE = []
H = np.eye(x_num*y_num*3)
H1 = (H.copy())[np.arange(x_num*y_num)*3]
H2 = (H.copy())[np.arange(x_num*y_num)*3+1]
H3 = (H.copy())[np.arange(x_num*y_num)*3+2]
h = np.arange(x_num*y_num)
R = np.eye(x_num*y_num)
P_f = np.eye(x_num*y_num*3)*0.4

set_condition(h_a, 4, 50)
#calc_observation(h_a, M_a, N_a, dx, dy, dt, g, D, tunami_equ2, steps)

msesh_x = np.arange(x_num)
msesh_y = np.arange(y_num)
#
f1 = open("data_t.txt", "r")
f2 = open("data_o.txt", "r")
h_t = get_data(f1, x_num, y_num)
h_a = get_data(f2, x_num, y_num)
h_a = np.random.normal(0, 1, [x_num, y_num])


#sys.exit()
for i in range(1, steps):
    h_f, M_f, N_f = runge_kutta4(tunami_equ2, h_a, M_a, N_a, dx, dy, dt, g, D)
    h_t = get_data(f1, x_num, y_num)
    x_a = (h_f.ravel())@H1 + (M_f.ravel())@H2 + (N_f.ravel())@H3
    y = get_data(f2)
    
    x_a = calc_3DVAR(x_a, y, H1, h, P_f, R)
    h_a = x_a[0::3].reshape(x_num, y_num)
    M_a = x_a[1::3].reshape(x_num, y_num)
    N_a = x_a[2::3].reshape(x_num, y_num)
    RMSE.append(np.linalg.norm(h_t- h_a)/np.sqrt(x_num*y_num))
    
    
    
    
    #具体的に波面がどうなっているのかを描写する
    if i%1==0:
        print(i)
#    #    imshow(h_a)
#    #    plt.ylim(-100,100)
#    #    plt.plot(h_a[:,x_num//2])
#    #    plt.show()
#        y = y.reshape(x_num, y_num)
#        plot3D(msesh_x, msesh_y, y, -30, 30, title="y")
        plot3D(msesh_x, msesh_y, h_a, -30, 30, title="h_a")
        plot3D(msesh_x, msesh_y, h_t, -30, 30, title="h_t")
#        plot3D(msesh_x, msesh_y, M_a+N_a, -300, 300, title="M_a")


#%%
RMSE = np.array(RMSE)
plt.ylim(0.3, 0.7)
plt.plot(RMSE)
plt.show()
print(RMSE.mean())
































#%%
print("running_time=", time.time()- time1)