# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:57:32 2020

@author: Dani
"""

# # # #
# Resolución EC. Vorticidad 2D:     D(omega_z)/Dt = 1/Re * Laplaciano(omega_z)
#                                   omega_z = -Laplaciano(psi)
#                                   u = d(psi)/dy;    v = -d(psi)/dx
# Método pseudo-espectral
# Esquema temporal: RK4
# ()_hat: coeficiente de Fourier
# # # #

####
# Vórtices modelados con distribución Gaussiana
# omega = gammaii*exp(rii*r^2)
# Long. característica: 1/sqrt(2*rii)
# Vel. caracerística: gammaii/sqrt(2*rii)
####

#Importar librerías
import matplotlib.pyplot as plt
from matplotlib import cm
from pylab import *
import math
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import apfft

# Control de la precisión con la librería mpmath
from mpmath import *
mp.dps = 4  # Establecer precisión decimal de longitud 8 dígitos 


# Datos ---
L = 2*math.pi  # Longitud del dominio
tfinal = 150
mu = 3.0e-6  # Viscosidad
CFL = 0.2  # CFL inicial
n = 2**9 #2**9  # Número de modos o puntos. Turbulencia 2D: eta = cte/Re^0.5
gammaii = 2  # Intensidad de los torbellinos
rii = 30  # Radio de los vórtices
Re = gammaii/(2*rii)/mu  # Reynolds basado en vórtices

# Mallado ---
dx = dy = L/n
x = linspace(0, L-(L/n), n)
y = linspace(0, L-(L/n), n)
x, y = np.meshgrid(x, y)

# Cálculo del CFL Viscoso ---
dtv = min([CFL*dx**2*Re, CFL*dy**2*Re])


# Inicializar Fourier (números de onda) ---
kx = 2*math.pi*fftfreq(len(x),L/len(x))
ky = 2*math.pi*fftfreq(len(y),L/len(y))
kx, ky = np.meshgrid(kx, ky)
kx = kx*2*math.pi/L
ky = ky*2*math.pi/L
poisson = Lap = -(kx**2+ky**2)  # Cálculo de la ecuación de Poisson en Fourier
poisson[0,0] = 1  # Punto singular de psi no definido


# Dealias a 2/3, equivalente al filtro en archivo ec-burgers-fft-rk4-2d-correcto.py ---
dealias = np.logical_and(abs(kx*L/(2*math.pi))<1/3*(n), abs(ky*L/(2*math.pi))<1/3*(n))


# Condición inicial ---
omega = L*0
nvx = 9
nvy = 9
for bucle1 in range(1,nvx):
    for bucle2 in range(1,nvy):
        if (bucle1==6 and bucle2==6):
            continue
        omega = omega + (-1)**(bucle1+bucle2)*gammaii*math.exp(-rii*((x-bucle1*L/nvx)**2 + (y-bucle2*L/nvy)**2))
        # omega = np.array(omega.tolist(),dtype=np.float64) para plotear


# Límites de colores en el plot y normalizado
vmin,vmax = -2,2
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
plt.show()


# Función para convertir un array mpc a mpf reales (elemento por elemento) ---
def realdani(matriz):
    for filas in range(0,n):
        for columnas in range(0,n):
            matriz[filas][columnas] = mp.re(matriz[filas,columnas])
    return(matriz)

# Funciones para convertir un array de mpf a float64 / a complex128
def mpmatrizafloat(matriz):
    matrizfloat = np.array(matriz.tolist(),dtype=np.float64)
    return matrizfloat
def mpmatrizacomplex(matriz):
    matrizfloat = np.array(matriz.tolist(),dtype=np.complex128)
    return matrizfloat

# Funciones para los cálculos ---
def rhs(du):    
    # Cálculos
    #   1. Dealias
    du = dealias*du
    
    #   2. Obtener psdi
    psi_hat = -du/poisson
    
    #   3. Obtener u,v
    u_hat = 1j*ky*psi_hat
    v_hat = -1j*kx*psi_hat
    
    #   4. Términos convectivos
    u = real(ifft2(u_hat))
    v = real(ifft2(v_hat))
    omega_x = real(ifft2(1j*kx*du))
    omega_y = real(ifft2(1j*ky*du))
    
    conv = u*omega_x + v*omega_y
    conv_hat = fft2(mpmatrizafloat(conv))
    
    #   5. Resultado
    return(1/Re*Lap*du-conv_hat)   
   
def calculou(du):
        # Cálculos
    #   1. Dealias
    du = dealias*du
    
    #   2. Obtener psdi
    psi_hat = -du/poisson
    
    #   3. Obtener u,v
    u_hat = 1j*ky*psi_hat
    
    #   5. Resultado
    return(u_hat)       
def calculov(dv):
        # Cálculos
    #   1. Dealias
    dv = dealias*dv
    
    #   2. Obtener psdi
    psi_hat = -dv/poisson
    
    #   3. Obtener u,v
    v_hat = -1j*kx*psi_hat
    
    #   5. Resultado
    return(v_hat)  
    

# Obtención de primeros valores en Fourier ---
omega_hat = fft2(omega)

# Obtener velocidades ---
psi_hat = -omega_hat/poisson
u_hat = 1j*ky*psi_hat
v_hat = -1j*kx*psi_hat
u = real(ifft2(u_hat))
v = real(ifft2(v_hat))

du = omega_hat
npaso = 0
ii = 0
t = 0

# Defino la listas que van a contener todos los valores de energía y enstrofia (normalizados sobre 1)
listaenergia = []
listaenstrofia = []

t1_start = time.process_time() 

# Bucle temporal ---
while npaso<2000:
    npaso = npaso + 1 # Número -de paso temporal
            
    # CFL convectivo en x
    dtx = CFL*dx/np.amax(u)
    # CFL convectivo en x
    dty = CFL*dy/np.amax(v)
    
    # Escoger el dt más restrictivo
    dt = min([dtv, dtx, dty])
    
    # RK4
    a1 = rhs(omega_hat)
    a2 = rhs(omega_hat + 0.5*dt*a1)
    a3 = rhs(omega_hat + 0.5*dt*a2)
    a4 = rhs(omega_hat + dt*a3)
    omega_hat = omega_hat*mpf('1.0') + (a1*mpf('1.0') + 2*a2*mpf('1.0') + 2*a3*mpf('1.0') + a4*mpf('1.0'))*dt*mpf('1.0')/6
    omega_hat = mpmatrizacomplex(omega_hat)
    u_hat = calculou(omega_hat)
    v_hat = calculov(omega_hat)
    t = t + dt
    
    # Guardar en archivos de texto la velocidad 
    if (mod(npaso,100)==0) or (npaso==1) :
        f = open("simulaciones/ump"+str(npaso)+".txt", "a")
        np.savetxt(f, real(ifft2(u_hat)))
        f.close()
        
        f = open("simulaciones/vmp"+str(npaso)+".txt", "a")
        np.savetxt(f, real(ifft2(v_hat)))
        f.close()
            
        
    print(npaso)
    t1_stop = time.process_time() 
    print("Tiempo transcurrido:", t1_stop-t1_start) 
    
    
 # Cálculo de la energía y la enstrofía total cada 100 pasos respecto a la inicial ---
if (mod(npaso,100)==0) or (npaso==1) :
    ktotal = sqrt(kx**2+ky**2) # Calculo el número de onda total para cada posición de la matriz
    E = 0.5*2*math.pi*(u_hat*np.conj(u_hat)+v_hat*np.conj(v_hat)) # Integral de perímetro de circunferencia (2D)
    Ens = 0.5*2*math.pi*(omega_hat*np.conj(omega_hat))
    
    Ektotal3d = dstack((ktotal,E)) # Superpongo las matrices de números de onda y energías sobre el eje 2
    Enstotal3d = dstack((ktotal,Ens))
    
    Ektotal2d = real(Ektotal3d.reshape((-1,2))) # Reconvierto la matriz a una nx2 (ktotal, Energía)
    Enstotal2d = real(Enstotal3d.reshape((-1,2)))
    
    Ektotal2dsorted = sorted(Ektotal2d,key=lambda x: x[0]) # Ordeno los valores por ktotal creciente
    Enstotal2dsorted = sorted(Enstotal2d,key=lambda x: x[0])
    
    # Procesamiento con pandas ---
    df = pd.DataFrame(Ektotal2dsorted, columns=['k', 'Energía']) # Convierto la matriz a un dataframe para el binning
    dfens = pd.DataFrame(Enstotal2dsorted, columns=['k','Enstrofía'])
    df['Enstrofía'] = dfens['Enstrofía']
    
    if npaso == 1:
        energiainicial = df['Energía'].sum()
        enstrofiainicial = df['Enstrofía'].sum()
        
    energiaactual = df['Energía'].sum()/energiainicial
    enstrofiaactual = df['Enstrofía'].sum()/enstrofiainicial
    
    listaenergia.append(energiaactual)
    listaenstrofia.append(enstrofiaactual)

