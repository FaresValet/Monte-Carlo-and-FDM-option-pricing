
import numpy as np
import matplotlib.pyplot as plt
# =============================================================================
# Heston call price using Monte-Carlo
# =============================================================================
def HestonCall(S0):
    Nmc=100
    K=10
    T=0.5
    r=0.1
    k=0.3
    p=0.7
    N=100
    Theta=0.3
    Etha=0.4
    v0=0.03
    dt=T/N
    V=np.zeros((N,Nmc))
    V[0,:]=v0
    S=np.zeros((N,Nmc))
    S[0,:]=S0
    Time=np.linspace(0,T,N)
    A=0
    for j in range(Nmc):
        for i in range(99):
           B=np.random.normal()
           Z=np.random.normal()
           V[i+1][j]=V[i][j]+k*(Theta-V[i][j])*dt+Etha*np.sqrt(V[i][j])*np.sqrt(dt)*B+0.25*Etha**2*dt*(B**2-1)
           S[i+1][j]=S[i][j]*np.exp((r-0.5*V[i][j])*dt+np.sqrt(V[i][j])*(p*np.sqrt(dt)*B+np.sqrt(1-p**2)*np.sqrt(dt)*Z))
        payoff=np.maximum(S[99][j]-K,0)
        A=A+payoff
        
    plt.plot(Time,V)  
    return np.exp(-r*(T))*A/Nmc


Start=np.linspace(0,25,100)
CallPrice=np.zeros(100)
for i in range(100):
    CallPrice[i]=HestonCall(Start[i])
    
plt.plot(Start,CallPrice)
