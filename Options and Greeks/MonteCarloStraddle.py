import numpy as np
import matplotlib.pyplot as plt
# =============================================================================
# Initializing our parameters
# =============================================================================
K=10
T=0.5
Nmc=1000
r =0.3 
sigma = 0.5 
T = 0.5 
S=np.linspace(0,40,100)
f=np.linspace(0,0.5,41) 
Straddle=np.zeros((100,41)) 
def straddlepayoff(S,K):
    if S>=K:
        return S-K
    else:
        return K-S
# =============================================================================
#  Straddle pricer with monte-carlo simulations   
# =============================================================================
def StraddlePrice(S0,Nmc,K,t)   :
    A=0
    for i in range(1,Nmc):
         ST=S0*np.exp((r-0.5*sigma**2)*(T-t)+sigma*np.sqrt(T-t)*np.random.normal())
         gain=straddlepayoff(ST,K)
         A=A+gain
    return np.exp(-r*(T-t))*A/Nmc

# =============================================================================
#    Array with straddle value depending on the strike and time
# =============================================================================
for i in range(0,100):
    for j in range(0,41):
        Straddle[i,j]=StraddlePrice(S[i],1000,10,f[j])
  


Stradle=np.transpose(Straddle)
fig = plt.figure()
ax = plt.axes(projection='3d')
x, y = np.meshgrid(S, f)
ax.plot_surface(x,y,Stradle)
plt.show()

ax.set_title('surface')