

import os
file_name = os.path.basename('C:\CYTechStudent/DeltaHedging.ext')
import numpy as np
from scipy.stats import norm 
import matplotlib.pyplot as plt
import math

# =============================================================================
# Delta Hedging with proportional transaction cost k0
# =============================================================================
def Call_BS(t,S,K,T,r,sigma):


    if t==T:
        return np.maximum(S-K,0)
    else: 
        d1=(np.log(S/K)+(r+0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
        d2=(np.log(S/K)+(r-0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
        return(S*norm.cdf(d1)-K*np.exp(-r*(T-t))*norm.cdf(d2))
    

def DeltaFunc(t,S,K,T,r,sigma):
    
    if t==T:
        return 1
    else:
        d1=(np.log(S/K)+(r+0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
        return norm.cdf(d1)
sigma=0.25
def Tradingfrequency(N):
    T=5
    S=[100]*N
    A=[0]*N
    dt=T/N
    t=np.linspace(0,T,N)
    r=0.05
    k0=0.01
    K=100
    B=[1]*N
    P=[0]*N
    Error=[0]*N
    Pact=[0]*N
    V=[0]*N
    V[0]=Call_BS(0,1,K,T,r,sigma)
    B[0]=V[0]-A[0]*S[0]-k0*abs(A[0])*S[0]
    A[0]=DeltaFunc(0,1,K,T,r,sigma)
    P[0]=A[0]*S[0]+B[0]
    Pact[0]=V[0]
    AugmentedSigma=sigma*np.sqrt(1+(k0/sigma)*np.sqrt(2/(dt*math.pi)))
    Abis=[0]*N
    Abis[0]=A[0]*S[0]
   
        
    
# =============================================================================
#     
# =============================================================================
    for i in range(0,N-1):
        
            S[i+1]=S[i]*np.exp((r-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*np.random.normal())
            A[i+1]=DeltaFunc(t[i+1],S[i+1],K,T,r,AugmentedSigma)
            B[i+1]=(A[i]-A[i+1])*S[i+1]+B[i]*(1+r*dt)-k0*abs((A[i]-A[i+1]))*S[i+1]
            P[i+1]=A[i+1]*S[i+1]+B[i+1]
            V[i+1] =Call_BS(t[i+1],S[i+1],K,T,r,AugmentedSigma)   
            Abis[i+1]=A[i+1]*S[i+1]
            Pact[i+1]=P[i+1]-(P[0]-V[0])*np.exp(r*t[i+1])
            Error[i+1]=Pact[i+1]-V[i+1]
    
    plt.plot(t,Abis,t,B)
    plt.xlabel("Time")
    plt.ylabel("Quantities")
    plt.title("Quantity of stock and quantity of cash")
    plt.show()
    print(A[0])
    """else:
            S[i+1]=S[i]*np.exp((r-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*np.random.normal())
            A[i+1]=A[i]
            B[i+1]=B[i]*(1+r*dt)
            P[i+1]=A[i+1]*S[i+1]+B[i+1]
            V[i+1] =Call_BS(t[i+1],S[i+1],K,T,r,AugmentedSigma) 
            Pact[i+1]=P[i+1]-(P[0]-V[0])*np.exp(r*t[i+1])
            Error[i+1]=Pact[i+1]-V[i+1]"""
          
    """plt.plot(t,Pact,t,V)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Option and replicating portfolio")
    plt.show()
    plt.plot(t,A,t,B)
    plt.xlabel("Time")
    plt.ylabel("Quantities")
    plt.title("Quantity of stock and quantity of cash")
    plt.show()
    
    plt.plot(t[1:N-1],Error[1:N-1])
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.title("Error of the replicating portfolio with delta hedging only")
    plt.show()"""
    
    return Pact[N-1]-V[N-1]

def multiplepaths(Nmc,N):
    Testori=[0]*Nmc
    alpha=0.1
    for i in range(Nmc):
      Testori[i]=Tradingfrequency(N)
    
    data=np.sort(Testori)
    Indexk=math.floor(Nmc*alpha)
    Var=data[Indexk]
    
    return Var,np.mean(Testori)


def minimum():
    
    Mali=[ (i*20) for i in range(1,50)]
    Vect=[0]*len(Mali)
    for i in range(len(Mali)):
       Vect[i]= multiplepaths(1000,Mali[i])
    
    plt.scatter(Mali,Vect)
    plt.title("Value at risk for different trading frequencies")
    plt.xlabel("Rebalance number")
    plt.ylabel("Value at risk")
    plt.show()
    max_value = max(Vect)
    max_index=Vect.index(max_value)
    return Mali[max_index]

"""def testor(mod):
    
  Test=[0]*100
  for i in range(100):
        Test[i]=np.var(Tradingfrequency(mod))
    
  return np.mean(Test)"""