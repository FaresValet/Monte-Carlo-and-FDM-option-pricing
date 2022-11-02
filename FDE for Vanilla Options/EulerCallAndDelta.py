
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Initial values
# =============================================================================
S0=5430.3
r=0.1
T=0.5
sigma=0.5
N=99
M=4999
L=20.0
Dt=T/(M+1)
Ds=L/(N+1)
K=10.0
S=np.linspace(0,20,N+2)
t=np.linspace(0,0.5,M+2)  
V=np.zeros((M+2,N+2))
Delta=np.zeros((M+2,N+2))
# =============================================================================
# Boundary conditions (Dirichlet) for explicit Euler method 
# =============================================================================
for i in range(0,N+2):
    V[M+1,i]=np.maximum(S[i]-K,0)
    
for n in range(0,M+1):
    V[n][0]=0
    V[n][N+1]=L-K*np.exp(-r*(T-t[n]))
   

# =============================================================================
# FDE Loop to calculate the estimate the PDE
# =============================================================================
for n in range(M+1,0,-1):
    for i in range(0,N+1):
        
        V[n-1,i]= V[n,i+1]*Dt*0.5*(sigma**2 *S[i]**2 /(Ds)**2 +r*(S[i])/(Ds)) +V[n,i]*(1-Dt*((sigma**2*S[i]**2 /(Ds)**2)+r))+V[n,i-1]*Dt/2 *((sigma**2*S[i]**2 )/(Ds)**2 -r*S[i]/Ds)

# =============================================================================
# Calculating Delta
# =============================================================================
for n in range (0,M+2):
    for i in range (1,N+1):
        Delta[n,i]=(V[n,i+1]-V[n,i-1])/(2*Ds)

    Delta[n,N+1]=Delta[n,N]

# =============================================================================
# Plotting Delta
# =============================================================================
plt.plot(S,Delta[1,:])
plt.ylim([0,1])
plt.show()
fig = plt.figure()
ax = plt.axes(projection='3d')
x, y = np.meshgrid(S, t)



ax.plot_surface(y, x, Delta)
ax.set_zlim3d(0, 1)
plt.show()

ax.set_title('surface')