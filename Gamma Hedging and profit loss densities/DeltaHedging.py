import os
file_name = os.path.basename('C:\CYTechStudent/DeltaHedging.ext')
import numpy as np
from scipy.stats import norm 
import matplotlib.pyplot as plt
# =============================================================================
# Regular Delta Hedging
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
sigma=0.5
T=5
N=200
S=[1]*N
A=[0]*N
dt=T/N
t=np.linspace(0,T,N)
r=0.05
K=1.5
B=[1]*N
P=[0]*N
Error=[0]*N
Pact=[0]*N
A[0]=DeltaFunc(0,1,K,T,r,sigma)
P[0]=A[0]*S[0]+B[0]
V=[0]*N
V[0]=Call_BS(0,1,K,T,r,sigma)
Nmc=10
W=[0]*N
W[0]=A[0]*S[0]/P[0]
for i in range(0,N-1):
    S[i+1]=S[i]*np.exp((r-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*np.random.normal())


for i in range(0,N-1):
    A[i+1]=DeltaFunc(t[i+1],S[i+1],K,T,r,sigma)
    B[i+1]=(A[i]-A[i+1])*S[i+1]+B[i]*(1+r*dt)
    P[i+1]=A[i+1]*S[i+1]+B[i+1]
    V[i+1] =Call_BS(t[i+1],S[i+1],K,T,r,sigma)   
    Pact[i+1]=P[i+1]-(P[0]-V[0])*np.exp(r*t[i+1])
    Error[i+1]=Pact[i+1]-V[i+1]
    W[i+1]=A[i+1]*S[i+1]/Pact[i+1]
Pact[0]=V[0]

plt.plot(t,Pact,t,V)
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Option and replicating portfolio")
plt.show()
plt.plot(t,A,t,B)
plt.xlabel("Time")
plt.ylabel("Quantities")
plt.title("Quantity of stock and quantity of cash")
plt.show()
plt.plot(t,W)
plt.title("W")
plt.show()
plt.plot(t,Error)
plt.xlabel("Time")
plt.ylabel("Error")
plt.title("Error of the replicating portfolio with delta hedging only")
plt.show()
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
