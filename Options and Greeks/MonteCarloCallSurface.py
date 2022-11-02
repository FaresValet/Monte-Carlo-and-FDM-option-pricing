import numpy as np
import matplotlib.pyplot as plt
# =============================================================================
# Initial parameters
# =============================================================================
K=10
T=0.5
Nmc=1000
r =0.3 
sigma = 0.5 
T = 0.5 
S=np.linspace(0,40,100)
f=np.linspace(0,0.5,41) 
Call=np.zeros((100,41)) 

# =============================================================================
# Call surface using Monte-Carlo simulations
# =============================================================================
def CallPrice(S0,Nmc,K,t)   :
    A=0
    for i in range(1,Nmc):
         ST=S0*np.exp((r-0.5*sigma**2)*(T-t)+sigma*np.sqrt(T-t)*np.random.normal())
         gain=np.maximum(ST-K,0)
         A=A+gain
    return np.exp(-r*(T-t))*A/Nmc

   
for i in range(0,100):
    for j in range(0,41):
        Call[i,j]=CallPrice(S[i],1000,10,f[j])
  


Cll=np.transpose(Call)
fig = plt.figure()
ax = plt.axes(projection='3d')
x, y = np.meshgrid(S, f)
ax.plot_surface(y,x,Cll)
plt.show()

ax.set_title('surface')