# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:25:52 2020

@author: Dani
"""
import numpy as np
from pylab import *
import matplotlib.pyplot as plt

listadifu = []
listadifv = []
listadifu.append(0)
listadifv.append(0)
for n in range(1,21):
    u1 = np.loadtxt(open("simulaciones/u"+str(100*n)+".txt", "rb"), delimiter=" ")
    v1 = np.loadtxt(open("simulaciones/v"+str(100*n)+".txt", "rb"), delimiter=" ")
    u1mp = np.loadtxt(open("simulaciones/ump"+str(100*n)+".txt", "rb"), delimiter=" ")
    v1mp = np.loadtxt(open("simulaciones/vmp"+str(100*n)+".txt", "rb"), delimiter=" ")
    udiferencia = u1mp - u1
    vdiferencia = v1mp - v1
    udiferenciacuadrado = udiferencia**2
    vdiferenciacuadrado = vdiferencia**2
    errorcuadu = sqrt(sum(udiferenciacuadrado))
    errorcuadv = sqrt(sum(vdiferenciacuadrado))
    listadifu.append(errorcuadu)
    listadifv.append(errorcuadv)
    print("Comparado lista "+str(n)+" de 20")
    
  
listadifua = []
listadifva = []
listadifua.append(0)
listadifva.append(0)
for n in range(1,21):
    u1a = np.loadtxt(open("simulaciones/ua"+str(100*n)+".txt", "rb"), delimiter=" ")
    v1a = np.loadtxt(open("simulaciones/va"+str(100*n)+".txt", "rb"), delimiter=" ")
    u1 = np.loadtxt(open("simulaciones/u"+str(100*n)+".txt", "rb"), delimiter=" ")
    v1 = np.loadtxt(open("simulaciones/v"+str(100*n)+".txt", "rb"), delimiter=" ")
    udiferencia = u1a - u1
    vdiferencia = v1a - v1
    udiferenciacuadrado = udiferencia**2
    vdiferenciacuadrado = vdiferencia**2
    errorcuadu = sqrt(sum(udiferenciacuadrado))
    errorcuadv = sqrt(sum(vdiferenciacuadrado))
    listadifua.append(errorcuadu)
    listadifva.append(errorcuadv)
    print("Comparado lista "+str(n)+" de 20")
    

    
# Plotear las gráficas de error en u y v
ejex = linspace(0,20,21)
plt.subplots(figsize=(14,8))
title("Comparación de la raíz del error cuadrático medio", size=20)
plt.xlabel('% de simulación', fontsize=19)
plt.ylabel('Raíz del error cuadrático medio', fontsize=19)
plot(100*ejex/21,listadifu,'b', label="RECM en u (4-16 dígitos)")
plot(100*ejex/21,listadifv,'r', label="RECM en v (4-16 dígitos)")
plot(100*ejex/21,listadifua,'bo', label="RECM en u (visc. modificada)")
plot(100*ejex/21,listadifva,'ro', label="RECM en v (visc. modificada)")
plt.tick_params(axis='both', labelsize=17)
plt.legend(loc="lower right", prop={'size': 15})
