import numpy as np
import matplotlib.pyplot as plt
# =============================================================================
# Initial Values
# =============================================================================
K=10
K2=5
T=0.5
Nmc=100
r =0.3 
sigma = 0.5 
T = 0.5 
T2=0.3
S=np.linspace(0,40,100)
f=np.linspace(0,0.5,41) 
Call=np.zeros((100,41)) 

# =============================================================================
# Call price MC
# =============================================================================
def CallPriceFunc(S0,Nmc,K,t)   :
    A=0
    for i in range(1,Nmc):
         ST=S0*np.exp((r-0.5*sigma**2)*(T-t)+sigma*np.sqrt(T-t)*np.random.normal())
         
         A=A+np.maximum(ST-K,0)
    return np.exp(-r*(T-t))*A/Nmc
# =============================================================================
# Call on call price function
# =============================================================================
def CompoundCallPrice(S0,Nmc,K,K2,t):
    A=0
    for i in range(1,Nmc):
         ST=S0*np.exp((r-0.5*sigma**2)*(T2-t)+sigma*np.sqrt(T2-t)*np.random.normal())
         Value=CallPriceFunc(ST,Nmc,K,T2)
         A=A+np.maximum(Value-K2,0)
    plt.plot(Value)
    return np.exp(-r*(T2-t))*A/Nmc
# =============================================================================
# Call on call plot
# =============================================================================
Start=np.linspace(0,25,100)
CallPrice=np.zeros(100)

for i in range(100):
    CallPrice[i]=CompoundCallPrice(Start[i],Nmc,K,K2,0)
    
plt.plot(Start,CallPrice)
    