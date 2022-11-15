# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 11:22:53 2022

@author: CYTech Student
"""

import numpy as np 
sigma=0.5
T=5
N=100
Nmc=10000
S=np.zeros((N,Nmc))
A=np.zeros((N,Nmc))
dt=T/N
t=np.linspace(0,T,N)
r=0.05
K=1.5
B=np.zeros((N,Nmc))
P=np.zeros((N,Nmc))
Error=np.zeros((N,Nmc))
Pact=np.zeros((N,Nmc))
V=np.zeros((N,Nmc))
Error=np.zeros((N,Nmc))
for i in range(Nmc):
    S[0][i]=1
Ct2=0
Vr2=0
# =============================================================================
# Hedging loop
# =============================================================================
for j in range(Nmc):
    for i in range(0,N-1):
        S[i+1][j]=S[i][j]*np.exp((r-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*np.random.normal())
    Ct2=Ct2 + S[N-1][j]
    Vr2=Vr2+S[N-1][j]**2

AssetEsp=Ct2/Nmc
EspAssetSquared=Vr2/Nmc
AssetVariance=EspAssetSquared-AssetEsp**2
TheoreticalMean=S[0][0]*np.exp((r-0.5*sigma**2)*T)*np.exp(0.5*T*sigma**2)
TheoreticalVariance=S[0][0]*np.exp((r-0.5*sigma**2)*2*T)*(np.exp(T*sigma**2)-1)*np.exp(T*sigma**2)
print('Final asset value mean is ', AssetEsp)
print('Final asset variance is', AssetVariance)
print('Theoretical asset value mean is ', TheoreticalMean)
print('Theoretical asset variance is', TheoreticalVariance)