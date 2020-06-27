# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:32:42 2020

@author: Dani
"""

# # # #
# Resolución EC. Burgers 2D:    u_t = mu*(u_xx + u_yy) - u*u_x - v*u_y
#                               v_t = mu*(v_xx + v_yy) - u*v_x - v*v_y. Por Fourier y RK4
#   mu = 1/Re
# # # #

import matplotlib.pyplot as plt
from matplotlib import cm
from pylab import *
import math
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from mpl_toolkits import mplot3d


L = math.pi
mu = 0.2
CFL = 0.05
n = 2**7 # Número de puntos de la malla
npaso = 16001 # Número de pasos temporales
x = linspace(-L, L-(L/n), n)
y = linspace(-L, L-(L/n), n)

# Malla 2D
x, y = np.meshgrid(x, y)
dx = dy = L/n

# Condiciones iniciales
u = sin(x)
v = sin(y)

u_hat = fft2(u)
v_hat = fft2(v)

# Números de onda y derivadas espectrales
kx = pi*fftfreq(len(x),L/len(x))
ky = pi*fftfreq(len(y),L/len(y))

kx, ky = np.meshgrid(kx, ky)
kx = 2*pi*kx/(2*L)
ky = 2*pi*ky/(2*L)
k1x = 1j*kx
k2x = k1x**2
k1y = 1j*ky
k2y = k1y**2

# Generamos una matriz cuadrada con las esquinas 1s y los valores centrales 0s. Nombre: filtroxy
unos = np.ones(int(np.ceil(n/3)))
ceros = np.zeros(int(np.floor(n/3)))
filax = np.concatenate((unos,ceros,unos), axis=0)[:n]
filay = np.concatenate((unos,ceros,unos), axis=0)[:n]
filtrox,filtroy = np.meshgrid(filax,filay)
filtroxy = floor((filtrox+filtroy)/2)

# Filtro más sencillo
dealias = np.logical_and(abs(kx*L/(2*pi))<1/3*(n), abs(ky*L/(2*pi))<1/3*(n))

# Límites de color del plot
vmin,vmax = -1,1
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

# Paso de tiempo viscoso
dtvx = CFL*dx**2/mu
dtvy = CFL*dy**2/mu
dtv = min(dtvx,dtvy)
    

plt.show()

for j in range(0,npaso):
    
    # Plot en el espacio físico
    u = real(ifft2(u_hat))
    v = real(ifft2(v_hat))
    
    # Paso de tiempo convectivo
    dtc = CFL*dx/max(np.amax(u),np.amax(v))

    dt = min(dtc, dtv)
        
    # Plot de la superficie
    if mod(j,2000)==0 :
        # surf = ax.plot_surface(x, y, u, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        # pcolormesh(x, y, v) #incluir ,norm=norm dentro del paréntesis para límites fijos de color
        fig = plt.figure(figsize=(14,10))
        ax = plt.axes(projection='3d')
        ax.contour3D(x, y, u, 50)
        ax.set_xlabel('x', fontsize=17)
        ax.set_ylabel('y', fontsize=17)
        ax.set_zlabel('u', fontsize=17)
        ax.set_xlim3d(-pi,pi)
        ax.set_ylim3d(-pi,pi)
        ax.set_zlim3d(-1,1)
        title('Componente u para paso '+str(j), size=20);
        plt.savefig('simulacionburgers2d/paso'+str(j)+'.png')
        plt.close()
        
        fig = plt.figure(figsize=(14,10))
        ax = plt.axes(projection='3d')
        ax.contour3D(x, y, v, 50)
        ax.set_xlabel('x', fontsize=17)
        ax.set_ylabel('y', fontsize=17)
        ax.set_zlabel('v', fontsize=17)
        ax.set_xlim3d(-pi,pi)
        ax.set_ylim3d(-pi,pi)
        ax.set_zlim3d(-1,1)
        title('Componente v para paso '+str(j), size=20);
        plt.savefig('simulacionburgers2d/paso'+str(j)+'v.png')
        plt.close()

    
    # Actualizamos función con RK4
    a1u = -k1x*fft2(0.5*u**2)*dealias - fft2(real(ifft2(k1y*u_hat))*v)*dealias + mu*(k2x + k2y)*u_hat
    a1v = -k1y*fft2(0.5*v**2)*dealias - fft2(real(ifft2(k1x*v_hat))*u)*dealias + mu*(k2x + k2y)*v_hat
    
    a2u = -k1x*fft2(0.5*(u + 0.5*dt*a1u)**2)*dealias - fft2(real(ifft2(k1y*fft2(u + 0.5*dt*a1u)))*(v + 0.5*dt*a1v))*dealias + mu*(k2x + k2y)*(u_hat + 0.5*dt*a1u)
    a2v = -k1y*fft2(0.5*(v + 0.5*dt*a1v)**2)*dealias - fft2(real(ifft2(k1x*fft2(v + 0.5*dt*a1v)))*(u + 0.5*dt*a1u))*dealias + mu*(k2x + k2y)*(v_hat + 0.5*dt*a1v)
    
    a3u = -k1x*fft2(0.5*(u + 0.5*dt*a2u)**2)*dealias - fft2(real(ifft2(k1y*fft2(u + 0.5*dt*a2u)))*(v + 0.5*dt*a2v))*dealias + mu*(k2x + k2y)*(u_hat + 0.5*dt*a2u)
    a3v = -k1y*fft2(0.5*(v + 0.5*dt*a2v)**2)*dealias - fft2(real(ifft2(k1x*fft2(v + 0.5*dt*a2v)))*(u + 0.5*dt*a2u))*dealias + mu*(k2x + k2y)*(v_hat + 0.5*dt*a2v)
    
    a4u = -k1x*fft2(0.5*(u + dt*a3u)**2)*dealias - fft2(real(ifft2(k1y*fft2(u + dt*a3u)))*(v + dt*a3v))*dealias + mu*(k2x + k2y)*(u_hat + dt*a3u)
    a4v = -k1y*fft2(0.5*(v + dt*a2v)**2)*dealias - fft2(real(ifft2(k1x*fft2(v + dt*a3v)))*(u + dt*a3u))*dealias + mu*(k2x + k2y)*(v_hat + dt*a3v)

    u_hat = u_hat + (a1u + 2*a2u + 2*a3u + a4u)*dt/6
    v_hat = v_hat + (a1v + 2*a2v + 2*a3v + a4v)*dt/6
    print(v.max())
    
    pause(0.001)    
