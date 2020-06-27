# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 15:38:30 2020

@author: Dani
"""

import matplotlib.pyplot as plt
from matplotlib import cm
from pylab import *
import math
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from operator import itemgetter
import pandas as pd

ktotal = sqrt(kx**2+ky**2) # Calculo el número de onda total para cada posición de la matriz
E = 0.5*2*pi*(u_hat*np.conj(u_hat)+v_hat*np.conj(v_hat)) # Integral de perímetro de circunferencia (2D)

Ektotal3d = dstack((ktotal,E)) # Superpongo las matrices de números de onda y energías sobre el eje 2
Ektotal2d = real(Ektotal3d.reshape((-1,2))) # Reconvierto la matriz a una nx2 (ktotal, Energía)
Ektotal2dsorted = sorted(Ektotal2d,key=lambda x: x[0]) # Ordeno los valores por ktotal creciente

# Procesamiento con pandas ---
df = pd.DataFrame(Ektotal2dsorted, columns=['k', 'Energía']) # Convierto la matriz a un dataframe para el binning
dfbins = df.groupby(pd.cut(df['k'], bins=512)).mean() # Agrupo en 'bins' y obtengo las medias en cada uno

# Plot de los bins en escala log-log para espectro de energía
dfbins.plot(x ='k', y='Energía', figsize=(16, 10), kind = 'line', linewidth=3)
title("Espectro de energía por números de onda", size=20)
plt.xlabel('k', fontsize=19)
plt.ylabel('Energía', fontsize=19)
plt.xlim([0.5,300])
plt.ylim([10**-8,10**8])
plt.tick_params(axis='both', labelsize=17)
plt.yscale('log')
plt.xscale('log')

# Cálculo de la energía total
ktotal = sqrt(kx**2+ky**2) # Calculo el número de onda total para cada posición de la matriz
E = 0.5*2*pi*(u_hat*np.conj(u_hat)+v_hat*np.conj(v_hat)) # Integral de perímetro de circunferencia (2D)
Ens = 0.5*2*pi*(omega_hat*np.conj(omega_hat))
Ektotal3d = dstack((ktotal,E)) # Superpongo las matrices de números de onda y energías sobre el eje 2
Enstotal3d = dstack((ktotal,Ens))
Ektotal2d = real(Ektotal3d.reshape((-1,2))) # Reconvierto la matriz a una nx2 (ktotal, Energía)
Enstotal2d = real(Enstotal3d.reshape((-1,2)))
Ektotal2dsorted = sorted(Ektotal2d,key=lambda x: x[0]) # Ordeno los valores por ktotal creciente
Enstotal2dsorted = sorted(Enstotal2d,key=lambda x: x[0])
# Procesamiento con pandas ---
df = pd.DataFrame(Ektotal2dsorted, columns=['k', 'Energía']) # Convierto la matriz a un dataframe para el binning

energiainicial = df['Energía'].sum()
energiaactual = df['Energía'].sum()/energiainicial

# Evolución de energía y enstrofía con el tiempo
plt.subplots(figsize=(14,8))
title("Evolución de energía y enstrofía con el tiempo", size=20)
plt.xlabel('% de simulación', fontsize=19)
plt.ylabel('E/E0, Ens/Ens0', fontsize=19)
plot(ejex/80,listaenergia/listaenergia[0], label="Energía")
plot(ejex/80,listaenstrofia/listaenstrofia[0], label="Enstrofía")
plt.tick_params(axis='both', labelsize=17)
plt.legend(loc="upper right", prop={'size': 15})