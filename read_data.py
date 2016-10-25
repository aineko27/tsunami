# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 01:06:47 2016

@author: seis
"""


import numpy as np
import matplotlib.pyplot as plt
from function_tunami import *

f1 = open("../../Fortran/tunami/data_t.txt")
f2 = open("../../Fortran/tunami/data_o.txt")
f3 = open("../../Fortran/tunami/data_a.txt")

RMSE01 = []
RMSE02 = []

x_num, y_num, steps = np.array(np.array(f1.readline()[2:-1].split(","), dtype=int))
msesh_x = np.arange(x_num)
msesh_y = np.arange(y_num)

for i in range(1, 10):
    h_t = get_data(f1, x_num, y_num)
    y = get_data(f2, x_num, y_num)
    h_a = get_data(f3, x_num, y_num)
    RMSE01.append(np.linalg.norm(y- h_t)/ np.sqrt(x_num*y_num))
    RMSE02.append(np.linalg.norm(h_t- h_a)/ np.sqrt(x_num*y_num))
    if i%10==0:
        print(i)
#        plot3D(msesh_x, msesh_y, y, -30, 30, title="h_o")
#        plot3D(msesh_x, msesh_y, h_a, -30, 30, title="h_a")
        plot3D(msesh_x, msesh_y, h_t, -30, 30, title="h_t")
        plot3D(msesh_x, msesh_y, y, -30, 30, title="y")
        plot3D(msesh_x, msesh_y, h_a, -30, 30, title="h_a")




f1.close()
f2.close()
RMSE01 = np.array(RMSE01)
plt.plot(RMSE01)
plt.show()
RMSE02 = np.array(RMSE02)
plt.plot(RMSE02)
plt.show()