import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import quad 
from ImpliedVolScipy import ImpliedVol
# =============================================================================
# Source paper: https://hal.sorbonne-universite.fr/hal-02273889/document
# Heston call price calculated using the characteristic function and integral form
# Calibration BS/Heston (Strange results)
# Note: the algorithm seems to "fail" and give negative prices when K is high
# =============================================================================
def hestonanalytical(phi,S0,v0,kappa,theta,sigma,rho,lambd,tau,r):
    #constants
    a=kappa*theta
    b=kappa+lambd
    #common terms
    rspi=rho*sigma*phi*1j
    #define d parameter given phi and b 
    d=np.sqrt((rho*sigma*phi*1j-b)**2 + (phi*1j+phi**2)*sigma**2)
    #define g parameter given phi b and d
    g=(b-rspi+d)/(b-rspi-d)
    #calculate characteristic function
    exp1=np.exp(r*phi*1j*tau)
    term2=S0**(phi*1j)*((1-g*np.exp(d*tau))/(1-g))**(-2*a/sigma**2)
    exp2=np.exp(a*tau*(b-rspi+d)/sigma**2+ v0*(b-rspi+d)*((1-np.exp(d*tau))/(1-g*np.exp(d*tau)))/sigma**2)
    return exp1*term2*exp2
def integrand(phi,S0,v0,kappa,theta,sigma,rho,lamb,tau,r):
    args=(S0,v0,kappa,theta,sigma,rho,lamb,tau,r)
    numerator=np.exp(r*tau)*hestonanalytical(phi-1j,*args)-K*hestonanalytical(phi,*args)
    denominator=1j*phi*K**(1j*phi)
    return numerator/denominator
def hestonpricer(S0,K,v0,kappa,theta,sigma,rho,lambd,tau,r):
    args=(S0,v0,kappa,theta,sigma,rho,lambd,tau,r)
    integral, err=np.real(quad(integrand,0,100,args=args))
    return (S0-K*np.exp(-r*tau))/2 + integral/np.pi


#test parameters
S0=100
K=100
v0=0.1
r=0.03
kappa=1.5768
theta=0.0398
sigma=0.3
lambd=0.575
rho=-0.5711
tau=1

print(hestonpricer(S0,K,v0,kappa,theta,sigma,rho,lambd,tau,r))

# =============================================================================
# Volatility smile attempt :)
# =============================================================================
strikes=np.arange(70,140,1)
calls=[]
for K in strikes:
  
    calls.append(hestonpricer(S0,K,v0,kappa,theta,sigma,rho,lambd,tau,r))

ivs = [ImpliedVol(C, S0, K, tau, r, type_ = 'call' ) for C, K in zip(calls,strikes)]

plt.plot(strikes, ivs)
plt.title('Weird plot')