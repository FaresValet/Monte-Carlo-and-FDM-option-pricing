
import numpy as np
import matplotlib.pyplot as plt
# =============================================================================
# Basis of Merton's model
# =============================================================================
N=100
T=1
mu=1
sigma=0.5
a=0.3
t=0
dt=T/N
Time=np.linspace(0,1,N)
Jump=[0.3,0.8]
MeanY=0.5
Newvec=np.sort(np.concatenate((Time,Jump)))
S=[10]*len(Newvec)
Y=[1]*len(Newvec)
for i in range(len(Newvec)):
    if Newvec[i] in Jump:
        Y[i]=(1+MeanY)*np.exp(-0.5*a**2+np.random.normal()*a)
for i in range(len(Newvec)-1):
    
    S[i+1]=S[i]*np.exp((mu-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*np.random.normal())*Y[i]

plt.plot(Newvec,S)