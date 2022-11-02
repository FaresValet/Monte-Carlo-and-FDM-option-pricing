import numpy as np
from scipy.stats import norm 
import matplotlib.pyplot as plt
# =============================================================================
# Vega surface plot
# =============================================================================
S0=5430.3
T=1
def Call_BS(t,S,K,T,r,sigma):
    d1=(np.log(S/K)+(r+0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
    d2=(np.log(S/K)+(r-0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))

    if t==T:
        return (np.max(S-K,0))
    else: 
        return(S*norm.cdf(d1)-K*np.exp(-r*(T-t))*norm.cdf(d2))
    

def Vega(t,S,K,T,r,sigma):
    d1=(np.log(S/K)+(r+0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
    return S*np.sqrt(T-t)/np.sqrt(2*np.pi)*np.exp(-d1**2/2)

L=20
S=np.linspace(0,20,101)
M=np.linspace(0,1,101)
Delta=np.zeros((101,101))
Ds=L/101
K=10

calltest=np.zeros((101,101))
vegatest=np.zeros((101,101))
    
for i in range (0,100):
    
    for j in range(0,100):
 
       
        calltest[i][j]=Call_BS(M[j], S[i], 10, 1, 0.1, 0.5)
        vegatest[i][j]=Vega(M[j], S[i], 10, 1, 0.1, 0.5)
    
       
for i in range (0,100):
    calltest[100][i]=L-K*np.exp(-0.1*(T-M[i]))
       
for i in range(0,100):
      calltest[i][100]=np.maximum(S[i]-10,0)

for n in range (0,101):
    for i in range (1,100):
        Delta[n,i]=(calltest[n,i+1]-calltest[n,i-1])/(2*Ds)

  
    Delta[n,0]=0
    Delta[n,100]=1 
fig = plt.figure()
ax = plt.axes(projection='3d')
x, y = np.meshgrid(M, S)

ax.plot_surface(x, y, vegatest)
ax.set_zlim3d(0, 4)
ax.view_init(50, 150)
plt.show()


