import numpy as np
import matplotlib.pyplot as plt
# =============================================================================
# Initial values for Crank-Nicolson's fde scheme
# 
# =============================================================================
r=0.05
T=1
sigma=0.5
N=299
M=4999
L=20
Dt=T/(M+1)
Ds=L/(N+1)

S=np.linspace(0,20,N+2)
t=np.linspace(0,1,M+2)  
V=np.zeros((M+2,N+2))
Q=np.zeros((M+2,N+2))
Qstar=np.zeros((M+2,N+2))
Delta=np.zeros((M+2,N+2))
Gamma=np.zeros((M+2,N+2))
Theta=np.zeros((M+2,N+2))
Vega=np.zeros((M+2,N+2))
A=[]
B=[]
D=[]
Dstar=[]
# =============================================================================
# Kron function for Thomas' algorithm
# =============================================================================
def kron(a,b):
    if a!=b:
        return 0
    else:
        return 1
# =============================================================================
#    Boundary conditions
# =============================================================================
for i in range(0,N+2):
    V[M+1,i]=np.maximum(S[i]-10,0)
# =============================================================================
#     Lists for Thomas' algorithm 
# =============================================================================
for i in range(0,N+1): 
    A.append(-0.25*Dt*(r*S[i]/Ds +(S[i]/Ds)**2 *sigma**2))
    B.append(0.25*Dt*(r*S[i]/Ds -(S[i]/Ds)**2 *sigma**2))
    D.append(1+0.5*Dt*((S[i]/Ds)**2 *sigma**2 +r))
    Dstar.append(0)
# =============================================================================
#     Boundary condition
# =============================================================================
for i in range(0,M+1):
    V[i,N+1]=L-10*np.exp(-r*(T-t[i]))
# =============================================================================
# Crank-Nicolson scheme and Thomas algorithm to solve the implicit scheme system
# =============================================================================
for n in range(M+1,0,-1):
    for i in range(0,N+1):
        Q[n,i] =V[n,i+1]*Dt*0.25*(sigma**2 *(S[i]/Ds)**2+r*S[i]/Ds)+V[n,i]*(1-0.5*Dt*(sigma**2*(S[i]/Ds)**2+r))+V[n,i-1]*Dt*0.25*(sigma**2*(S[i]/Ds)**2-r*S[i]/Ds)-kron(N,i)*A[N]*V[n-1,N+1]-kron(1,i)*B[1]*V[n-1,0]


    Dstar[1]=D[1]
    Qstar[n,1]=Q[n,1]
    for i in range(2,N+1):
        Dstar[i]=D[i]-B[i]*A[i-1]/Dstar[i-1]
        Qstar[n,i]=Q[n,i]-B[i]*Qstar[n,i-1]/Dstar[i-1]


    V[n-1,N]=Qstar[n,N]/Dstar[N]
    for i in range (N-1,0,-1):
        V[n-1,i]=(Qstar[n,i]-A[i]*V[n-1,i+1])/Dstar[i]

# =============================================================================
# Calculating Delta and Gamma
# =============================================================================

for n in range (0,M+2):
    for i in range (1,N+1):
        Delta[n,i]=(V[n,i+1]-V[n,i-1])/(2*Ds)
        Gamma[n,i]=(V[n,i+1]+V[n,i-1]-2*V[n,i])/(Ds)**2
     
    Delta[n,N+1]=Delta[n,N]



fig = plt.figure()
ax = plt.axes(projection='3d')
x, y = np.meshgrid(S, t)



ax.plot_surface(y, x, Delta)
ax.set_zlim3d(0, 1)
ax.set_ylim3d(0, 20)


