import numpy as np
from scipy.stats import norm 
import matplotlib.pyplot as plt
import seaborn as sns
# =============================================================================
# Hedging using the given implied volatility 
# =============================================================================
def StochEin():
    if np.random.binomial(1,0.8)==1:
        return 0.3
    else:
        return 0.5

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
sigma=0.4
T=5
N=100
Nmc=1
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
for i in range(Nmc):
    S[0][i]=1
    B[0][i]=1
    A[0][i]=DeltaFunc(0,1,K,T,r,sigma)
    P[0][i]=A[0][i]*S[0][i]+B[0][i]
    V[0][i]=Call_BS(0,1,K,T,r,sigma)


Ct=0
Vr=0
for j in range(Nmc):
    for i in range(0,N-1):
        sigmah=StochEin()
        S[i+1][j]=S[i][j]*np.exp((r-0.5*sigmah**2)*dt+sigmah*np.sqrt(dt)*np.random.normal())


    for i in range(0,N-1):
        A[i+1][j]=DeltaFunc(t[i+1],S[i+1][j],K,T,r,sigma)
        B[i+1][j]=(A[i][j]-A[i+1][j])*S[i+1][j]+B[i][j]*(1+r*dt)
        P[i+1][j]=A[i+1][j]*S[i+1][j]+B[i+1][j]
        V[i+1][j] =Call_BS(t[i+1],S[i+1][j],K,T,r,sigma)   
        Pact[i+1][j]=P[i+1][j]-(P[0][j]-V[0][j])*np.exp(r*t[i+1])
      
    Pact[0][j]=V[0][j]
    Ct=Ct + Pact[N-1][j]-V[N-1][j]
    Vr=Vr + (Pact[N-1][j]-V[N-1][j])**2
Esp=Ct/Nmc
EspCar=Vr/Nmc
Variance=EspCar-Esp**2
data=np.sort(Pact[N-1][:]-V[N-1][:])
y = np.arange(len(data)) / (len(data) - 1)

#plot CDF
plt.plot(t,Pact,t,V)
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Option and replicating portfolio with stochastic implied volatility")
plt.show()
plt.plot(data, y)
plt.xlabel('data')
plt.show()
sns.distplot(data, hist=False)
plt.show()