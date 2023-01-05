# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math 

# =============================================================================
# Local Volatility Model 
# =============================================================================

def DupirePrice(a,b,ag,mg,bg,pg,S0,r,T_max,K_max,M,N,h,method='CEV'):
    Time=np.linspace(0,T_max,M+2)
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
    if method=='CEV':
        def LocalVol(i,h):
            Sigma=a/K[i]**b +h
            return Sigma
    elif method=='Gatheral':
        def LocalVol(i,h):
            Sigma=bg*(pg*(K[i]-mg)+np.sqrt((K[i]-mg)**2 + ag**2)) +h
            return Sigma
        
    for i in range(N+2):
       V[0,i]=np.maximum(S0-K[i],0)
      
    for n in range(1,M+2):
        V[n,0]=S0
        V[n,N+1]=0
    
    for n in range(M+1):
        for i in range(1,N+1):
            A[i]=0.25*dt*(r*K[i]/dk-LocalVol(i,h)**2 *(K[i]/dk)**2)
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
           
    
    
    
    return V
    
def DupireVega(a,b,ag,mg,bg,pg,h,voltype):
    T_max=0.5
    K_max=20
    M=49
    N=199
    T=np.linspace(0,T_max,M+2)
    K=np.linspace(0,K_max,N+2)
    Vega=(DupirePrice(a,b,ag,mg,bg,pg,10,0.1,0.5,20,49,199,h,method=voltype)-DupirePrice(a,b,ag,mg,bg,pg,10,0.1,0.5,20,49,199,0,voltype))/h
    return Vega
def UsefulDupirePrices(a,b,ag,mg,bg,pg,S0,r,T_max,K_max,M,N,h,voltype):
    if voltype=='CEV':
        Strikes=np.arange(7,14.5,0.5)
    elif voltype=='Gatheral':
        Strikes=np.arange(5,19,1)
    S0,r,T_max,K_max,M,N,h=10,0.1,0.5,20,49,199,0
    V=DupirePrice(a,b,ag,mg,bg,pg,S0,r,T_max,K_max,M,N,h,method=voltype)
    p=np.zeros((len(Strikes))).astype(int)
    dk=K_max/(N+1)
    for i in range(len(Strikes)):
        p[i]=Strikes[i]/dk
        

    return V[50,p],p

def UsefulVegaPrices(a,b,ag,mg,bg,pg,S0,r,T_max,K_max,M,N,h,voltype):
    S0,r,T_max,K_max,M,N,h=10,0.1,0.5,20,49,199,0.01
    Vega=DupireVega(a,b,ag,mg,bg,pg,h,voltype)
    _,p=UsefulDupirePrices(a,b,ag,mg,bg,pg,S0,r,T_max,K_max,M,N,h,voltype)
    return Vega[50,p]

def LevenbergMarquardt(S0,r,T_max,K_max,M,N,epsilon,lamb):
    Strikes=np.arange(7,14.5,0.5)
    MarketPrices=np.array([3.3634,2.9092,2.4703,2.0536,1.6666,1.3167,1.0100,0.7504,0.5389,0.3733,0.2491,0.1599,0.0986,0.0584,0.0332])
    a,b,ag,mg,bg,pg,S0,r,T_max,K_max,M,N,epsilon,lamb=1,1,1,1,1,1,10,0.1,0.5,20,49,199,10**(-6),10**(-3)
    d=[1,1]
    res=np.zeros(len(Strikes))
    J=np.zeros((len(Strikes),2))
    while np.linalg.norm(d,2)>epsilon:
        Vdupire,_=UsefulDupirePrices(a,b,ag,mg,bg,pg,S0,r,T_max,K_max,M,N,0,'CEV')
        Vega=UsefulVegaPrices(a,b,ag,mg,bg,pg,S0,r,T_max,K_max,M,N,0.01,'CEV')
        for i in range(len(Strikes)):
            res[i]=MarketPrices[i]-Vdupire[i]
            J[i,0]=-Vega[i]/(Strikes[i]**b)
            J[i,1]=Vega[i]*(np.log(Strikes[i])*a)/(Strikes[i]**b)
       
        M=(np.dot(J.T,J)+lamb*np.identity(2))
        d=-np.dot(np.linalg.inv(M),J.T).dot(res.reshape((len(Strikes),1)))
        a+=d[0]
        b+=d[1]
        print(a)
        print(res)
    
    return a,b,np.linalg.norm(res,2)
def LevenbergMarquardtGatheral(S0,r,T_max,K_max,M,N,epsilon,lamb):
    StrikesGatheral=np.arange(5,19,1)
    MarketPricesGatheral=[5.2705,4.3783,3.5510,2.8138,2.1833,1.6651,1.2541,0.9374,0.6983,0.5195,0.3851,0.2817,0.1987,0.1277]
    a,b,ag,mg,bg,pg,S0,r,T_max,K_max,M,N,epsilon,lamb=1,1,1,1,0.05,0.1,10,0.1,0.5,20,49,199,10**(-6),10**(-3)
    d=[1,1]
    res=np.zeros(len(StrikesGatheral))
    J=np.zeros((len(StrikesGatheral),2))
    while np.linalg.norm(d,2)>epsilon:
        Vdupire,_=UsefulDupirePrices(a,b,ag,mg,bg,pg,S0,r,T_max,K_max,M,N,0,'Gatheral')
        Vega=UsefulVegaPrices(a,b,ag,mg,bg,pg,S0,r,T_max,K_max,M,N,0.01,'Gatheral')
        for i in range(len(StrikesGatheral)):
            res[i]=MarketPricesGatheral[i]-Vdupire[i]
            J[i,0]=-Vega[i]*bg*ag/(np.sqrt((StrikesGatheral[i]-mg)**2 + ag**2))
            J[i,1]=-Vega[i]*bg*(-pg+(mg-StrikesGatheral[i])/(np.sqrt((StrikesGatheral[i]-mg)**2 + ag**2)))
        print(res)
        M=(np.dot(J.T,J)+lamb*np.identity(2))
        d=-np.dot(np.linalg.inv(M),J.T).dot(res.reshape((len(StrikesGatheral),1)))
        ag+=d[0]
        mg+=d[1]
    return ag,mg,np.linalg.norm(res,2)

"""plt.plot(K,V[30,:])
fig = plt.figure()
ax = plt.axes(projection='3d')
x, y = np.meshgrid(K, Time)
ax.plot_surface(y, x, V,cmap='viridis')
ax.view_init(50, 0)
ax.set_title('surface')
plt.show()"""
""" plt.plot(K,Vega[30,:])
 plt.show()
 fig = plt.figure()
 ax = plt.axes(projection='3d')
 x, y = np.meshgrid(K, T)
 ax.plot_surface(y, x, Vega,cmap='viridis')
 ax.view_init(50, 50)
 ax.set_title('surface')
 plt.show()"""