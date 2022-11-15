import os
file_name = os.path.basename('C:\CYTechStudent/DeltaHedging.ext')
import numpy as np
from scipy.stats import norm 
import matplotlib.pyplot as plt
import seaborn as sns
import math
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
# Delta function and stochastic volatilities
# =============================================================================
def DeltaFunc(t,S,K,T,r,sigma):
    
    if t==T:
        return 1
    else:
        d1=(np.log(S/K)+(r+0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
        return norm.cdf(d1)
def Model1():
    if np.random.binomial(1,0.8)==1:
        return 0.3
    else:
        return 0.5
def Model2(i):
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
#     Deltahedging function
# =============================================================================
def Deltahedge(Nmc,tr,alpha):
    
    """sigma=0.5""" # Remove quotations if no stochasticVol
    T=5
    N=100
    
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
    # Counters for expectation and variance
    # =============================================================================
    Ct=0
    Vr=0
    Ct2=0
    Vr2=0
    # =============================================================================
    # Hedging loop, in this guess we simulate the stochastic volatility following model 2, one can get the constant volatility case just by setting sigma to a constant value and removing or setting to comments the unecessary coding
    # =============================================================================
    for j in range(Nmc):
        if np.random.binomial(1,0.5)==1:
          Sigmastart=0.5
        else:
          Sigmastart=0.3
        
        S[0][j]=1
        B[0][j]=1
        A[0][j]=DeltaFunc(0,1,K,T,r,Sigmastart)
        P[0][j]=A[0][j]*S[0][j]+B[0][j]
        V[0][j]=Call_BS(0,1,K,T,r,Sigmastart)
            
        sigma=Model2(Sigmastart)
    
        for i in range(0,N-1):
            sigma=Model2(sigma)
            if i%tr==0:
                S[i+1][j]=S[i][j]*np.exp((r-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*np.random.normal())
                A[i+1][j]=DeltaFunc(t[i+1],S[i+1][j],K,T,r,sigma)
                B[i+1][j]=(A[i][j]-A[i+1][j])*S[i+1][j]+B[i][j]*(1+r*dt)
                P[i+1][j]=A[i+1][j]*S[i+1][j]+B[i+1][j]
                V[i+1][j] =Call_BS(t[i+1],S[i+1][j],K,T,r,sigma)   
                Pact[i+1][j]=P[i+1][j]-(P[0][j]-V[0][j])*np.exp(r*t[i+1])
                Error[i+1][j]=Pact[i+1][j]-V[i+1][j]
            else:
                S[i+1][j]=S[i][j]*np.exp((r-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*np.random.normal())
                A[i+1][j]=A[i][j]
                B[i+1][j]=B[i][j]*(1+r*dt)
                P[i+1][j]=A[i+1][j]*S[i+1][j]+B[i+1][j]
                V[i+1][j] =Call_BS(t[i+1],S[i+1][j],K,T,r,sigma)   
                Pact[i+1][j]=P[i+1][j]-(P[0][j]-V[0][j])*np.exp(r*t[i+1])
            
          
        Pact[0][j]=V[0][j]
        Ct=Ct + Pact[N-1][j]-V[N-1][j]
        Vr=Vr + (Pact[N-1][j]-V[N-1][j])**2
        Ct2=Ct2 + S[N-1][j]
        Vr2=Vr2+S[N-1][j]**2
    Esp=Ct/Nmc
    EspCar=Vr/Nmc
    Variance=EspCar-Esp**2
    AssetEsp=Ct2/Nmc
    EspAssetSquared=Vr2/Nmc
    AssetVariance=EspAssetSquared-AssetEsp**2
    TheoreticalMean=S[0][0]*np.exp((r-0.5*sigma**2)*T)*np.exp(0.5*T*sigma**2)
    TheoreticalVariance=S[0][0]*np.exp((r-0.5*sigma**2)*2*T)*(np.exp(T*sigma**2)-1)*np.exp(T*sigma**2)
    # =============================================================================
    # Sorting the data to get the cdf
    # =============================================================================
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
    plt.plot(S)
    plt.title("Multiple asset paths In the second model")
    plt.xlabel('Time')
    plt.ylabel('Asset Value')
    plt.show()
    plt.plot(t,Pact,c='g',label='actualized portfolio')
    plt.plot(t,V,c='r', label='Option Value')
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Option and replicating portfolio")
    plt.show()
    plt.plot(t,A,t,B)
    plt.xlabel("Time")
    plt.ylabel("Quantities")
    plt.title("Quantity of stock and cash")
    plt.show()
    plt.plot(t,Error)
    plt.xlabel("Time")
    plt.ylabel("P&L")
    plt.title("P&L Error")
    plt.show()
    plt.plot(data, y)
    plt.xlabel("P&L")
    plt.ylabel("Value")
    plt.title("Cumulative distribution function in the second model")
    plt.show()
    sns.distplot(data, hist=False)
    plt.xlabel("P&L")
    plt.ylabel("Value")
    plt.title("Probability density function in the second model")
    plt.show()
    print('Final asset value mean is ', AssetEsp)
    print('Final asset variance is', AssetVariance)
    print('Theoretical asset value mean is ', TheoreticalMean)
    print('Theoretical asset variance is', TheoreticalVariance)
    print('P&L mean ', Esp )
    print('P&L Variance', Variance)
    return Var,Esp,Variance,data,y,Indexk
# =============================================================================
# For the probability densities with different trading frequencies
# =============================================================================
"""P0=Deltahedge(100,100,0.1)
P=Deltahedge(100,10,0.1)
P2=Deltahedge(100,4,0.1)
P3=Deltahedge(100,2,0.1)
P4=Deltahedge(100,1,0.1)
sns.distplot(P0,hist=False,label=' once')
sns.distplot(P,hist=False,label=' 10 dt')
sns.distplot(P2,hist=False,label=' 4 dt')
sns.distplot(P3,hist=False,label=' 2 dt')
sns.distplot(P4,hist=False,label=' all the time')
plt.title('Probability densities for different rebalancement frequencies')
plt.xlim(-0.8,0.8)
plt.legend()
plt.show()"""
# =============================================================================
# For the mean-variance as a function of the trading frequency
# =============================================================================
"""Var,Esp,Variance=Deltahedge(400,100,0.1)
Var1,Esp1,Variance1=Deltahedge(400,10,0.1)
Var2,Esp2,Variance2=Deltahedge(400,4,0.1)
Var3,Esp3,Variance3=Deltahedge(400,2,0.1)
Var4,Esp4,Variance4=Deltahedge(400,1,0.1)
Freq=[100,50,25,10,1]
Esperance=[Esp4,Esp3,Esp2,Esp1,Esp]
Variance=[Variance4,Variance3,Variance2,Variance1,Variance]
plt.plot(Freq,Esperance)
plt.title("Mean as a function of the trading frequency")
plt.xlabel("Frequency meaning the number of times we hedge")
plt.ylabel("Mean")
plt.show()
plt.plot(Freq,Variance)
plt.title("Variance as a function of the trading frequency")
plt.xlabel("Frequency meaning the number of times we hedge")
plt.ylabel("Variance")
plt.show()"""
# =============================================================================
# For the value at risk
# =============================================================================
"""Var,Esp,Variance,data,y,Indexk=Deltahedge(400,1,0.1)
Var1,Esp1,Variance1,data1,y1,Indexk1=Deltahedge(400,10,0.1)
print("The value at risk if we rebalance every dt is", Var)
print("The value at risk if we rebalance once every 10*dt is", Var1)
plt.plot(data,y,c="y",label="CDF")
plt.hlines(0.1,data[0],data[399])
plt.text(0.1, 0.15, ' alpha=0.1', ha='left', va='center')
plt.title("Cumulative distribution function when hedging all the time")
idx = np.argwhere(np.diff(np.sign(y - 0.1)))
plt.plot(data[idx], y[idx], 'o')
plt.vlines(data[Indexk], -0.1, 0.1, linestyles='--')
plt.legend()
plt.show()
plt.plot(data1,y1,c="r",label="CDF")
plt.hlines(0.1,data1[0],data1[399])
plt.text(0.2, 0.15, ' alpha=0.1', ha='left', va='center')
idx = np.argwhere(np.diff(np.sign(y1 - 0.1)))
plt.plot(data1[idx], y1[idx], 'o')
plt.vlines(data1[Indexk1], -0.1, 0.1, linestyles='--')
plt.legend()
plt.title("Cumulative distribution function when hedging once every 10dt")
plt.show()
print("The value at risk if we rebalance every dt is", Var)
print("The value at risk if we rebalance once every 10*dt is", Var1)"""