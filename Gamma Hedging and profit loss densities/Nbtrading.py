import os
file_name = os.path.basename('C:\CYTechStudent/DeltaHedging.ext')
import numpy as np
from scipy.stats import norm 
import matplotlib.pyplot as plt
import seaborn as sns
import math
# =============================================================================
# """Call"""
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
# =============================================================================
# "Stochastic" Volatility taking two values in the first model with prob 0.8/0.2
# and transition probabilities in the second model
# =============================================================================
def StochEin():
    if np.random.binomial(1,0.8)==1:
        return 0.3
    else:
        return 0.5
def SigmaStochZwei(i):
    if i == 0.5: 
        if np.random.binomial(1,0.05) == 1:
            return 0.3
        else:
            return 0.5
    if i == 0.3: 
        if np.random.binomial(1,0.05) == 1:
            return 0.5
        else:
            return 0.3
   
# =============================================================================
# Hedging function with the number of times we hedge as parameter    
# =============================================================================
def NTrading(tr):
    alpha=0.1
    T=5
    N=100
    Nmc=1000
    S=np.zeros((N,Nmc))
    A=np.zeros((N,Nmc))
    dt=T/N
    t=np.linspace(0,T,N)
    r=0.05
    K=1.5
    B=np.zeros((N,Nmc))
    P=np.zeros((N,Nmc))
   
    Pact=np.zeros((N,Nmc))
    V=np.zeros((N,Nmc))
    Sigma2=[0]*N
    a= 0.5
    for i in range(100):
        Sigma2[i]=SigmaStochZwei(a)
        a =  Sigma2[i]
        
       
    for i in range(Nmc):
        S[0][i]=1
        B[0][i]=1
        A[0][i]=DeltaFunc(0,1,K,T,r,StochEin())
        P[0][i]=A[0][i]*S[0][i]+B[0][i]
        V[0][i]=Call_BS(0,1,K,T,r,StochEin())
    
    
    Ct=0
    Vr=0
    for j in range(Nmc):
        for i in range(0,N-1):
            S[i+1][j]=S[i][j]*np.exp((r-0.5*StochEin()**2)*dt+StochEin()*np.sqrt(dt)*np.random.normal())
        
        
        
        for i in range (0,N-1):
            if  (i+1)%tr==0:
              A[i+1][j]=DeltaFunc(t[i+1],S[i+1][j],K,T,r,StochEin())
              B[i+1][j]=(A[i][j]-A[i+1][j])*S[i+1][j]+B[i][j]*(1+r*dt)
              P[i+1][j]=A[i+1][j]*S[i+1][j]+B[i+1][j]
              V[i+1][j] =Call_BS(t[i+1],S[i+1][j],K,T,r,StochEin())   
              Pact[i+1][j]=P[i+1][j]-(P[0][j]-V[0][j])*np.exp(r*t[i+1])
            else:
               
               A[i+1][j]=A[i][j]
               B[i+1][j]=B[i][j]*(1+r*dt)
               P[i+1][j]=A[i+1][j]*S[i+1][j]+B[i+1][j]
               V[i+1][j] =Call_BS(t[i+1],S[i+1][j],K,T,r,StochEin())   
               Pact[i+1][j]=P[i+1][j]-(P[0][j]-V[0][j])*np.exp(r*t[i+1])
            
      
          
        Pact[0][j]=V[0][j]
        Ct=Ct + Pact[N-1][j]-V[N-1][j]
        Vr=Vr + (Pact[N-1][j]-V[N-1][j])**2
    Esp=Ct/Nmc
    EspCar=Vr/Nmc
    Variance=EspCar-Esp**2
    data=np.sort(Pact[N-1][:]-V[N-1][:])
    y = np.arange(len(data)) / (len(data) - 1)
    Indexk=math.floor(Nmc*alpha)
    Var=data[Indexk]
  
    Test1=[0]*N
    Quantile1=[0]*N
    Quantile2=[0]*N
    part=np.linspace(0,1,N)
    for i in range(N):
        Test1[i]=np.random.normal(Esp,np.sqrt(Variance))
    
    for i in range(N):
        Quantile1[i]=np.quantile(Test1,part[i])
        Quantile2[i]=np.quantile(data,part[i])
    #plot CDF
    plt.plot(data,y)
    plt.title('Cumulative distribution function for modulo '+str(tr))
    plt.show()
    plt.plot(t,Sigma2)
    plt.title("Sigmastoch")
    plt.show()
    sns.distplot(data)
    plt.title('Probability density function in the first volatility model for modulo '+str(tr))
    plt.show()
    plt.plot(t,V,t,Pact)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title('Option and replicating portfolio for modulo'+str(tr))
    plt.show()
    plt.plot(t,S)
    plt.xlabel("Time")
    plt.ylabel("Value of the underlying")
    plt.title("Underlying asset paths for the second volatility model")
    plt.show()
    plt.plot(Quantile1,Quantile2)
    plt.plot(Quantile1,Quantile1)
    plt.show()
    print(Quantile1)
    print(Quantile2)
    print(Var)
   
    return Sigma2
# =============================================================================
# Probability density functions if we hedge every dt, then every 2dt, every 4dt
# etc
# =============================================================================
"""P=NTrading(100)
P2=NTrading(50)
P3=NTrading(25)
P4=NTrading(1)
sns.distplot(P,hist=False)
sns.distplot(P2,hist=False)
sns.distplot(P3,hist=False)
sns.distplot(P4,hist=False)
plt.xlim(-1,1)
plt.show()"""
