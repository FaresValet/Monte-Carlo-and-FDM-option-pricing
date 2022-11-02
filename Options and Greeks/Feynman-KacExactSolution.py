# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 13:47:50 2022

@author: CYTech Student
"""

import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import norm 
import matplotlib.pyplot as plt

S0=5430.3
r=0.05
T=4/12

"""sigma=np.sqrt(2*np.abs((np.log(S0/K)+r*T)/T))"""
z=np.random.standard_normal(100)
"""S=S0*np.exp((r-0.5*Sigma**2)*T+Sigma*np.sqrt(T)*z)"""
def Call_BS(t,S,K,T,r,sigma):
    d1=(np.log(S/K)+(r+0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
    d2=(np.log(S/K)+(r-0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))

    if t==T:
        return (np.max(S-K,0))
    else: 
        return(S*norm.cdf(d1)-K*np.exp(-r*(T-t))*norm.cdf(d2))
    

def Vega(t,S,K,T,r,sigma):
    d1=(np.log(S/K)+(r+0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
    return S*np.sqrt(T-t)/np.sqrt(2*np.pi)*np.exp(-d1**2/2)
  
L=[None]*100
S=[None]*100
calltest=[None]*100
vegatest=[None]*100

for i in range(1,100):
 
    S[i]=0.2*i
    calltest[i]=Call_BS(0, S[i], 10, 1, 0.1, 0.5)
    vegatest[i]=Vega(0,S[i],10,1,0.1,0.5)
    L[i]=np.maximum(S[i]-10,0)
 

def F(marche,t,S,K,T,r,sigma):
    return Call_BS(t, S, K, T, r, sigma)-marche
t=0
K=[5125, 5225, 5325, 5425, 5525, 5625, 5725,5825]
    
M=[475,405,340,280.5,226,179.5,139,105]
sigma=[]
for i in range(0,8):
    sigma.append(np.sqrt(2*np.abs(np.log(S0/K[i])+r*T)/T))


for i in range(0,8):
    
   
    while abs(F(M[i],t,S0,K[i],T,r,sigma[i]))>0.000001:
        sigma[i]=sigma[i]-(F(M[i],t,S0,K[i],T,r,sigma[i])/Vega(t,S0,K[i],T,r,sigma[i]))
     
      
print(sigma)

plt.plot(K,sigma)
plt.show()