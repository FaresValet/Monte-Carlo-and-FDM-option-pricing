import os
file_name = os.path.basename('C:\CYTechStudent/DeltaHedging.ext')
import numpy as np
from scipy.stats import norm 
import matplotlib.pyplot as plt
import seaborn as sns
# =============================================================================
# Call function
# =============================================================================
def Call_BS(t,S,K,T,r,sigma):


    if t==T:
        return np.maximum(S-K,0)
    else: 
        d1=(np.log(S/K)+(r+0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
        d2=(np.log(S/K)+(r-0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
        return(S*norm.cdf(d1)-K*np.exp(-r*(T-t))*norm.cdf(d2))
    
# =============================================================================
# Delta function
# =============================================================================
def DeltaFunc(t,S,K,T,r,sigma):
    
    if t==T:
        return 1
    else:
        d1=(np.log(S/K)+(r+0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
        return norm.cdf(d1)
# =============================================================================
#     Initializing arrays and parameters 
# =============================================================================
sigma=0.5
T=5
N=100
Nmc=100
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
# =============================================================================
# Initial portfolio A=Delta and B=Cash
# =============================================================================
for i in range(Nmc):
    S[0][i]=1
    B[0][i]=1
    A[0][i]=DeltaFunc(0,1,K,T,r,sigma)
    P[0][i]=A[0][i]*S[0][i]+B[0][i]
    V[0][i]=Call_BS(0,1,K,T,r,sigma)

# =============================================================================
# Counters for expectation and variance
# =============================================================================
Ct=0
Vr=0
# =============================================================================
# Hedging loop
# =============================================================================
for j in range(Nmc):
    for i in range(0,N-1):
        S[i+1][j]=S[i][j]*np.exp((r-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*np.random.normal())


    for i in range(0,N-1):
        A[i+1][j]=DeltaFunc(t[i+1],S[i+1][j],K,T,r,sigma)
        B[i+1][j]=(A[i][j]-A[i+1][j])*S[i+1][j]+B[i][j]*(1+r*dt)
        P[i+1][j]=A[i+1][j]*S[i+1][j]+B[i+1][j]
        V[i+1][j] =Call_BS(t[i+1],S[i+1][j],K,T,r,sigma)   
        Pact[i+1][j]=P[i+1][j]-(P[0][j]-V[0][j])*np.exp(r*t[i+1])
        Error[i+1][j]=Pact[i+1][j]-V[i+1][j]
      
    Pact[0][j]=V[0][j]
    Ct=Ct + Pact[N-1][j]-V[N-1][j]
    Vr=Vr + (Pact[N-1][j]-V[N-1][j])**2
Esp=Ct/Nmc
EspCar=Vr/Nmc
Variance=EspCar-Esp**2
# =============================================================================
# Sorting the data to get the cdf
# =============================================================================
data=np.sort(Pact[N-1][:]-V[N-1][:])
y = np.arange(len(data)) / (len(data) - 1)

#plot CDF
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
plt.plot(t,Error)
plt.xlabel("Time")
plt.ylabel("P&L")
plt.title("P&L Error")
plt.show()
plt.plot(data, y)
plt.xlabel("P&L")
plt.ylabel("Value")
plt.title("Cumulative distribution function")
plt.show()
sns.distplot(data, hist=False)
plt.xlabel("P&L")
plt.ylabel("Value")
plt.title("Probability density function")
plt.show()