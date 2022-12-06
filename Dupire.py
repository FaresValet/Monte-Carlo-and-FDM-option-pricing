# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math 

# =============================================================================
# Modèle à volatilité locale
# =============================================================================

def DupirePrice(a,b,S0,r,T_max,K_max,M,N,h):
    T=np.linspace(0,T_max,M+2)
    K=np.linspace(0,K_max,N+2)
    A=np.zeros(N+2)
    D=np.zeros(N+2)
    Dstar=np.zeros(N+2)
    B=np.zeros(N+2)
    V=np.zeros((M+2,N+2))
    C=np.zeros((M+2,N+2))
    Cstar=np.zeros((M+2,N+2))
    dt=T_max/(M+1)
    dk=K_max/(N+1)
    def kron(a,b):
        if a!=b:
            return 0
        else:
            return 1
    def LocalVol(i,h):
        Sigma=a/K[i]**b +h
        return Sigma
    for i in range(N+2):
       V[0,i]=np.maximum(S0-K[i],0)
      
    for n in range(1,M+2):
        V[n,0]=S0
        V[n,N+1]=0
    
    for n in range(M+1):
        for i in range(1,N+1):
            A[i]=0.25*dt*(r*K[i]/dk-LocalVol(i,h)**2 * (K[i]/dk)**2)
            D[i]=(1+0.5*dt*LocalVol(i,h)**2*(K[i]/dk)**2)
            B[i]=-0.25*dt*(r*K[i]/dk+LocalVol(i,h)**2 * (K[i]/dk)**2)
            C[n,i]=-A[i]*V[n,i+1]+(1-0.5*dt*LocalVol(i,h)**2*(K[i]/dk)**2)*V[n,i] -B[i]*V[n,i-1]-kron(i,1)*B[1]*S0
    
        Dstar[1]=D[1]
        Cstar[n,1]=C[n,1]
        for i in range(2,N+1):
            Dstar[i]=D[i]-B[i]*A[i-1]/Dstar[i-1]
            Cstar[n,i]=C[n,i]-B[i]*Cstar[n,i-1]/Dstar[i-1]
        
        V[n+1,N]=Cstar[n,N]/Dstar[N]
        for i in range(N-1,0,-1):
            V[n+1,i]=(Cstar[n,i]-A[i]*V[n+1,i+1])/Dstar[i]
            print(i)
    plt.plot(K,V[30,:])
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x, y = np.meshgrid(K, T)
    ax.plot_surface(y, x, V,cmap='viridis')
    ax.view_init(50, 0)
    ax.set_title('surface')
    plt.show()
    
    
    return V
    
def DupireVega(a,b,h):
    T_max=0.5
    K_max=20
    M=49
    N=199
    T=np.linspace(0,T_max,M+2)
    K=np.linspace(0,K_max,N+2)
    Vega=(DupirePrice(1,1,10,0.1,0.5,20,49,199,h)-DupirePrice(1,1,10,0.1,0.5,20,49,199,0))/h
    plt.plot(K,Vega[30,:])
    plt.show()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x, y = np.meshgrid(K, T)
    ax.plot_surface(y, x, Vega,cmap='viridis')
    ax.view_init(50, 50)
    ax.set_title('surface')
    plt.show()
    