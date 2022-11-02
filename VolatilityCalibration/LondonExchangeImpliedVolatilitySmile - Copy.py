
import numpy as np
from scipy.stats import norm 
import matplotlib.pyplot as plt
# =============================================================================
# Initialization of our parameters
# =============================================================================
S0=5430.3
r=0.05
T=4/12

# =============================================================================
# Call option function
# =============================================================================
def Call_BS(t,S,K,T,r,sigma):
    d1=(np.log(S/K)+(r+0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
    d2=(np.log(S/K)+(r-0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))

    if t==T:
        return (np.max(S-K,0))
    else: 

        return(S*norm.cdf(d1)-K*np.exp(-r*(T-t))*norm.cdf(d2))
    
# =============================================================================
# Function to calculate Vega, the derivative of the call with respect to sigma
# =============================================================================
def Vega(t,S,K,T,r,sigma):
    d1=(np.log(S/K)+(r+0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
    return S*np.sqrt(T-t)/np.sqrt(2*np.pi)*np.exp(-d1**2/2)
# =============================================================================
#  Lists to calculate the value of the call and vega for arbitrary values 
# =============================================================================
L=[None]*100
S=[None]*100
calltest=[None]*100
vegatest=[None]*100

for i in range(1,100):
 
    S[i]=0.2*i
    calltest[i]=Call_BS(0, S[i], 10, 1, 0.1, 0.5)
    vegatest[i]=Vega(0,S[i],10,1,0.1,0.5)
    L[i]=np.maximum(S[i]-10,0)
 
# =============================================================================
# This function gives the difference between the market and BS price 
# =============================================================================
def Function(marche,t,S,K,T,r,sigma):
    P=Call_BS(t, S, K, T, r, sigma)-marche
    return P
# =============================================================================
# Strikes and market prices for vol calibration
# =============================================================================
t=0
K=[5125, 5225, 5325, 5425, 5525, 5625, 5725,5825]
    
M=[475,405,340,280.5,226,179.5,139,105]
# =============================================================================
# Initializing vol values to make sure Newton's algorithm converges
# =============================================================================
sigma=[]
for i in range(0,8):
    sigma.append(np.sqrt(2*np.abs(np.log(S0/K[i])+r*T)/T))

# =============================================================================
# Newton's algorithm
# =============================================================================
for i in range(0,8):
    
   
    while abs(Function(M[i],t,S0,K[i],T,r,sigma[i]))>0.0001:
        sigma[i]=sigma[i]-(Function(M[i],t,S0,K[i],T,r,sigma[i])/Vega(t,S0,K[i],T,r,sigma[i]))
     
      
# =============================================================================
# Printing the difference to make sure it converges
# =============================================================================
for i in range(8):
    print(Function(M[i],t,S0,K[i],T,r,sigma[i]))
# =============================================================================
#     Vol smile
# =============================================================================
    
plt.plot(K,sigma)
plt.xlabel("K")
plt.ylabel("\u03C3")
plt.title("Volatility Smile")
plt.show()