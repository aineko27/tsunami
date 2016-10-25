# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 14:08:34 2016

@author: seis
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_data(file, x_num=0, y_num=0):
    if(x_num!=0):
        return np.array(file.readline().strip().split(),dtype=np.float64).reshape(y_num, x_num)
    else:
        return np.array(file.readline().strip().split(),dtype=np.float64)
#def get_data(file, x_num=0, y_num=0):
#    if(x_num!=0):
#        return np.array(file.readline()[:-1].split("   ")[1:], dtype=float).reshape(y_num, x_num)
#    else:
#        return np.array(file.readline()[:-1].split("   ")[1:], dtype=float)

#imshowを適当に定義しとく
def imshow(M, v_max=0, v_min=0):
    if v_max==v_min:
        plt.colorbar(plt.imshow(M, interpolation="nearest"))
    else:
        plt.colorbar(plt.imshow(M, interpolation="nearest", vmax=v_max, vmin=v_min))
    plt.show()

#3Dでグラフを描く
def plot3D(x, y, Z, zmin=0, zmax=0, title=""):
    ax = Axes3D(plt.figure())
    if zmax!=zmin:
        ax.set_zlim3d(zmin, zmax)
    X, Y = np.meshgrid(x, y)
    ax.plot_wireframe(X, Y, Z)
    if title!="":
        plt.title(title)
    plt.show()

#津波の時間発展方程式。こっちは周期境界条件になっているバージョン
def tunami_equ1(h_a, M_a, N_a, dx, dy, dt, g, D):
    a = np.arange(h_a.shape[1])
    b = np.arange(h_a.shape[0])
    dh = -(M_a[:,a+1-len(a)]- 2*M_a+ M_a[:,a-1])/ dx**2 - (N_a[b+1-len(b),:]- 2*N_a+ N_a[b-1,:])/ dy**2
    dM = -g*D*(h_a[:,a+1-len(a)]- 2*h_a+ h_a[:,a-1])/ dx**2
    dN = -g*D*(h_a[b+1-len(b),:]- 2*h_a+ h_a[b-1,:])/ dy**2
    return dh*dt, dM*dt, dN*dt
    
#津波の時間発展方程式。こっちは固定端(?)になっているバージョン
def tunami_equ2(h_a, M_a, N_a, dx, dy, dt, g, D):
    m = h_a.shape[1]
    n = h_a.shape[0]
    dh = np.zeros([n,m])
    dM = np.zeros([n,m])
    dN = np.zeros([n,m])
    dh[1:n-1,1:m-1] = -(M_a[1:n-1,2:m]- 2*M_a[1:n-1,1:m-1]+ M_a[1:n-1,0:m-2])/dx**2- (N_a[2:n,1:m-1]- 2*N_a[1:n-1,1:m-1]+ N_a[0:n-2,1:m-1])/dy**2
    dM[1:n-1,1:m-1] = -g*D[1:n-1,1:m-1]*(h_a[1:n-1,2:m]- 2*h_a[1:n-1,1:m-1]+ h_a[1:n-1,0:m-2])/ dx**2
    dN[1:n-1,1:m-1] = -g*D[1:n-1,1:m-1]*(h_a[2:n,1:m-1]- 2*h_a[1:n-1,1:m-1]+ h_a[0:n-2,1:m-1])/ dy**2
    return dh*dt, dM*dt, dN*dt
    
def tunami_equ3(h_a, M_a, N_a, dx, dy, dt, g, D):
    h_a = h_a.reshape(h_a.shape(0), -1, 1)
    m = h_a.shape[1]
    n = h_a.shape[0]
    dh = np.zeros([n,m,1])
    dM = np.zeros([n,m,1])
    dN = np.zeros([n,m,1])
    dh[1:n-1,1:m-1] = -(M_a[1:n-1,2:m]- 2*M_a[1:n-1,1:m-1]+ M_a[1:n-1,0:m-2])/dx**2- (N_a[2:n,1:m-1]- 2*N_a[1:n-1,1:m-1]+ N_a[0:n-2,1:m-1])/dy**2
    dM[1:n-1,1:m-1] = -g*D[1:n-1,1:m-1]*(h_a[1:n-1,2:m]- 2*h_a[1:n-1,1:m-1]+ h_a[1:n-1,0:m-2])/ dx**2
    dN[1:n-1,1:m-1] = -g*D[1:n-1,1:m-1]*(h_a[2:n,1:m-1]- 2*h_a[1:n-1,1:m-1]+ h_a[0:n-2,1:m-1])/ dy**2


#4次のルンゲクッタで近似する
def runge_kutta4(equ, h_a, M_a, N_a, dx, dy, dt, g, D):
    h1, M1, N1 = equ(h_a, M_a, N_a, dx, dy, dt, g, D)
    h2, M2, N2 = equ(h_a+ h1/2, M_a+ M1/2, N_a+ N1/2, dx, dy, dt, g, D)
    h3, M3, N3 = equ(h_a+ h2/2, M_a+ M2/2, N_a+ N2/2, dx, dy, dt, g, D)
    h4, M4, N4 = equ(h_a+ h3, M_a+ M3, N_a+ N3, dx, dy, dt, g, D)
    return h_a+ (h1+ 2*h2+ 2*h3+ h4)/6, M_a+ (M1+ 2*M2+ 2*M3+ M4)/6, N_a+ (N1+ 2*N2+ 2*N3+ N4)/6

#波の初期条件を設定する
def set_condition(Mtrx, sig, val):
    for i in range(Mtrx.shape[1]):
        for j in range(Mtrx.shape[0]):
            Mtrx[i, j] = np.exp(-((i-(Mtrx.shape[1]-1)/2)**2 + (j-(Mtrx.shape[0]-1)/2)**2)/(2*sig*sig))*val

def calc_observation(Mtrx1, Mtrx2, Mtrx3, dx, dy, dt, g, D, equ, steps):
    f1 = open("data_t.txt", "wb")
    f2 = open("data_o.txt", "wb")
    for i in range(steps):
        Mtrx1, Mtrx2, Mtrx3 = runge_kutta4(equ, Mtrx1, Mtrx2, Mtrx3, dx, dy, dt, g, D)
        np.savetxt(f1, Mtrx1.reshape(1,-1))
        np.savetxt(f2, (Mtrx1+ np.random.normal(0, 1, Mtrx1.shape)).reshape(1,-1))
        
    f1.close()
    f2.close()
    

def calc_3DVAR(x, y, H, h, B, R):
    W = sp.linalg.solve((R+ B[0::3,:][:,0::3]).T, B[:,0::3].T, sym_pos=True, overwrite_a=True, overwrite_b=True, check_finite=False).T
    return x+ W@ (y-x[0::3])






















