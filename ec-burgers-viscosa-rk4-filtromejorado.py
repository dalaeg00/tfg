# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 23:46:58 2020

@author: Dani
"""
# # # #
# Resolución EC. Burgers viscosa: u_t = -u*u_x + mu*u_xx. Por Fourier y RK4
# # # #

# El primer cambio a introducir es sustituir u*u_x por (0.5*u^2)_x
# 
from pylab import *
import math
import time

L = math.pi
CFL = 0.05
mu = 0.01
n = 2**9 # Número de puntos de la malla
Nt = 16000 # Número de pasos temporales
x = linspace(-L, L, n)
dx = 2*L/(n-1)

# condición inicial y asignación de valores iniciales a la función
u_i = cos(x)
u = np.ones(n)*u_i
u_conv = np.zeros(2*n)

u_hat = fft(u)
#Mínimo y máximo para extremos del plot
u_min = min(u)
u_max = max(u)
k1 = 1j*pi*fftfreq(len(x),L/len(x))
k2 = k1**2

# Creamos una lista de onda cuadrada de longitud n y con ceros en {n/3, 2*n/3}
unos = np.ones(int(np.ceil(n/3)))
ceros = np.zeros(int(np.floor(n/3)))
filtro = np.concatenate((unos,ceros,unos), axis=0)

# Paso de tiempo viscoso
dtv = CFL*dx**2/mu;

t1_start = time.process_time() 
for j in range(1,Nt):
    # Volvemos al espacio físico
    u = real(ifft(u_hat))
    
    # Paso de tiempo convectivo
    dtc = CFL*dx/max(u);
    
    dt = min(dtc, dtv)

    # Actualizamos función con RK4
    # Extiendo el vector a 3*n/2 puntos (Zero-padding, de-aliasing, Orszag) y completo con ceros delante y detrás
    # Hay que añadir int(n*0.25)) ceros delante y detrás del vector
    # Por último, se toman solo los valores entre n/4 y 5*n/4
    a1 = -k1*fft(0.5*u**2)*filtro[:n] + mu*k2*u_hat
    a2 = -k1*fft(0.5*(u + 0.5*dt*a1)**2)*filtro[:n] + mu*k2*(u_hat + 0.5*dt*a1)
    a3 = -k1*fft(0.5*(u + 0.5*dt*a2)**2)*filtro[:n] + mu*k2*(u_hat + 0.5*dt*a2)
    a4 = -k1*fft(0.5*(u + dt*a3)**2)*filtro[:n] + mu*k2*(u_hat + dt*a3)
    u_hat = u_hat + (a1 + 2*a2 + 2*a3 + a4)*dt/6
    
    pause(0.0001)    

t1_stop = time.process_time() 
print("Tiempo transcurrido en la simulación:", t1_stop-t1_start) 

# Plot en el espacio físico
plt.clf()
plt.subplots(figsize=(14,8))
plot(x, u,'*')
title("Paso "+str(Nt), size=20)
plt.xlabel('x', fontsize=17)
plt.ylabel('u(x)', fontsize=17)
show()
ylim([u_min, u_max])


# ---------------------- NO EJECUTAR ---------------------
# Para hacer plot de frecuencias
freq = linspace(-L,L,n)
f = cos(x)
f_hat = fft(f)
plot(freq,abs(f_hat/x))