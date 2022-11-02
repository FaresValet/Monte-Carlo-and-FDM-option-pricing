# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 18:49:07 2022

@author: CYTech Student
"""

import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import norm 
import matplotlib.pyplot as plt
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D

Nmc=100
L=20
T=0.5
r=0.1
k=0.3
p=0.7
N=100
Theta=0.3
Etha=0.4
v0=0.03
sigma=0.5
N=99
M=4999
L=20.0
Dt=T/(M+1)
Ds=L/(N+1)
K=10.0
S=np.linspace(0,20,N+2)
t=np.linspace(0,0.5,M+2)  
V=np.zeros((M+2,N+2))
Delta=np.zeros((M+2,N+2))
for i in range(0,N+2):
    V[M+1,i]=np.maximum(S[i]-K,0)
    
for n in range(0,M+1):
    V[n][0]=0
    V[n][N+1]=L-K*np.exp(-r*(T-t[n]))
   


for n in range(M+1,0,-1):
    for i in range(0,N+1):
        
        V[n-1,i]= V[n,i+1]*Dt*0.5*(sigma**2 *S[i]**2 /(Ds)**2 +r*(S[i])/(Ds)) +V[n,i]*(1-Dt*((sigma**2*S[i]**2 /(Ds)**2)+r))+V[n,i-1]*Dt/2 *((sigma**2*S[i]**2 )/(Ds)**2 -r*S[i]/Ds)

