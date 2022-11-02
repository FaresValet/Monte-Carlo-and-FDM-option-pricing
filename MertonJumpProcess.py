import matplotlib.pyplot as plt
import numpy as np 
from scipy.stats import norm
# =============================================================================
# Call BS function
# =============================================================================
def Call_BS(t,S,K,T,r,sigma):
    d1=(np.log(S/K)+(r+0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
    d2=(np.log(S/K)+(r-0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))

    if t==T:
        return (np.max(S-K,0))
    else: 

        return(S*norm.cdf(d1)-K*np.exp(-r*(T-t))*norm.cdf(d2))
# =============================================================================
# Function to get the asset path with jumps following a homogenous poisson dist 
# =============================================================================
def MertonProcess(S,T,r,sigma,lamb,m,v,N,Nmc):
    size=(N,Nmc)
    dt=T/N
    Poisson=np.multiply(np.random.poisson(lamb*dt,size=size),np.random.normal(m,v,size=size)).cumsum(axis=0)
    Geometric=np.cumsum(((r-0.5*sigma**2-lamb*(m+v**2*0.5)*dt+sigma*np.sqrt(dt)*np.random.normal(size=size))),axis=0)
    return np.exp(Geometric+Poisson)*S0
# =============================================================================
# Closed-form solution to the call price if assets have jumps
# =============================================================================
def CallMerton(S,K,T,r,sigma,m,v,lamb):
    p=0
    for i in range(40):
        r_i=r-lamb*(m-1)+(i*np.log(m))/T
        sigma_i=np.sqrt(sigma**2+(i*v**2)/T)
        Factorial=np.math.factorial(i)
        p+=(np.exp(-m*lamb*T)*(m*lamb*T)**i/(Factorial))*Call_BS(0,S,K,T,r_i,sigma_i)
    return p
# =============================================================================
# Parameters    
# =============================================================================
S0=100 
T=1
r=0.02
m=0
v=0.3
lamb=1
N=255
Nmc=200000
sigma=0.2
K=100
# =============================================================================
# Option price with monte-carlo, closed-form and regular BS, we notice the price
# =============================================================================
# is higher if the asset has random jumps because it adds uncertainty so it raises
# the premium
# =============================================================================
# =============================================================================
Merton=MertonProcess(S0, T, r, sigma, lamb, m, v, N, Nmc)
mcprice = np.maximum(Merton[-1]-K,0).mean() * np.exp(-r*T) 
cfprice=CallMerton(S0,K,T,r,sigma,np.exp(m+0.5*v**2),v,lamb)
BSPrice=Call_BS(0,S0,K,T,r,sigma)
print('Merton Price =', cfprice)
print('Monte Carlo Merton Price =', mcprice)
print('Black Scholes Price =', BSPrice)
"""plt.plot(Merton)
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.title('Merton Jump Process')"""