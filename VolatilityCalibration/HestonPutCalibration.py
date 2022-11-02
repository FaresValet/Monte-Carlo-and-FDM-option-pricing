import numpy as np
import matplotlib.pyplot as plt
from ImpliedVolScipy import ImpliedVol
# =============================================================================
# Initial parameters
# =============================================================================
Nmc=5000
L=20
T=1
r=0.02
k=4
p=-0.9
N=2000
Theta=0.02
Etha=0.9
v0=0.02
dt=T/N
S0=100
# =============================================================================
# Function to get the asset value with stochastic vol (Heston Model)
# =============================================================================
def hestonpaths(S0,T,k,Theta,v0,p,Etha,N,Nmc,returnvol=False):
    size=(Nmc,N)
    price=np.zeros(size)
    sigmas=np.zeros(size)
    S_t=S0
    v_t=v0
    mu=np.array([0,0])
    cov=np.array([[1,p],[p,1]])
    for t in range(N):
        W=np.random.multivariate_normal(mu,cov,size=Nmc)*np.sqrt(dt)
        S_t=S_t*(np.exp((r-0.5*v_t)*dt+np.sqrt(v_t)*W[:,0]))
        v_t=np.abs(v_t+k*(Theta-v_t)*dt+Etha*np.sqrt(v_t)*W[:,1])
        price[:,t]=S_t
        sigmas[:,t]=v_t
    if returnvol==True:
        return price, sigmas
    else:
        return price
# =============================================================================
#     Getting the last value S_T
# =============================================================================
price = hestonpaths(100,1,4,0.02,0.02,-0.9,0.9,2000,5000,returnvol=False)[:,-1]
# =============================================================================
# Strikes and put option price valuation using MC
# =============================================================================
strikes=np.arange(30,200,1)
puts=[]
for K in strikes:
    P=np.mean(np.maximum(K-price,0))*np.exp(-r*T)
    puts.append(P)

# =============================================================================
# Finding the volatility such that BS-put=Heston-put pricewise and plotting the smile
# =============================================================================
ivs = [ImpliedVol(P, S0, K, T, r, type_ = 'put' ) for P, K in zip(puts,strikes)]

plt.plot(strikes, ivs)