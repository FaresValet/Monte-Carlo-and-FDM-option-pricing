import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
# =============================================================================
# Heston call price using Monte-Carlo
# =============================================================================
def HestonCall(S0,Nmc,K,T,r,k,p,N,Theta,Etha,v0):
    dt=T/N
    V=np.zeros((N,Nmc))
    V[0,:]=v0
    Vsym=np.zeros((N,Nmc))
    Vsym[0,:]=v0
    S=np.zeros((N,Nmc))
    S[0,:]=S0
    Ssym=np.zeros((N,Nmc))
    Ssym[0,:]=S0
    Time=np.linspace(0,T,N)
    A=0
    M=0
    LogReturn=np.zeros(Nmc)
    for j in range(Nmc):
        for i in range(N-1):
           B=np.random.normal()
           Z=np.random.normal()
           V[i+1][j]=V[i][j]+k*(Theta-V[i][j])*dt+Etha*np.sqrt(V[i][j])*np.sqrt(dt)*B+0.25*Etha**2*dt*(B**2-1)
           S[i+1][j]=S[i][j]*np.exp((r-0.5*V[i][j])*dt+np.sqrt(V[i][j])*(p*np.sqrt(dt)*B+np.sqrt(1-p**2)*np.sqrt(dt)*Z))
           Vsym[i+1][j]=Vsym[i][j]+k*(Theta-Vsym[i][j])*dt+Etha*np.sqrt(Vsym[i][j])*np.sqrt(dt)*-B+0.25*Etha**2*dt*(B**2-1)
           Ssym[i+1][j]=Ssym[i][j]*np.exp((r-0.5*Vsym[i][j])*dt+np.sqrt(Vsym[i][j])*(p*np.sqrt(dt)*-B+np.sqrt(1-p**2)*np.sqrt(dt)*-Z))
        payoffUs=np.maximum(S[-1][j]-K,0)
        payoff=np.maximum(S[-1][j]-K,0)+np.maximum(Ssym[-1][j]-K,0)
        LogReturn[j]=np.log(S[-1][j]/S0)
        A+=payoff
        M+=payoffUs
    CallRed=0.5*np.exp(-r*(T))*A/Nmc
    Call=np.exp(-r*(T))*M/Nmc
    return CallRed,Call,LogReturn,V,S
S0,Nmc,K,T,r,k,p,N,Theta,Etha,v0=1,1000,1,0.5,0.01,2,-0.9,100,0.04,0.3,0.04
Values=[-0.9,0,0.9]
for i in range(3):
    S0,Nmc,K,T,r,k,p,N,Theta,Etha,v0=1,1000,1,0.5,0.01,2,Values[i],100,0.04,0.3,0.04
    _,_,LogReturn,V,S=HestonCall(S0,Nmc,K,T,r,k,Values[i],N,Theta,Etha,v0)
    sns.kdeplot(LogReturn,label='rho={}'.format(Values[i]))
plt.legend() 
plt.title('LogReturn Distributions rho=0,0.9,-0.9')
plt.show()
plt.plot(V)  
plt.title('Stochastic vol')
plt.show()
plt.plot(S)
plt.title('Heston Asset Price')
plt.show()
Start=np.linspace(0.01,20,100)
CallPriceReduced=np.zeros(100)
CallPrice=np.zeros(100)
for i in range(100):
    CallRed,Call,_,_,_=HestonCall(Start[i],100,10,0.5,0.01,2,-0.9,100,0.04,0.3,0.04)
    CallPriceReduced[i]=CallRed
    CallPrice[i]=Call
    
plt.plot(Start,CallPrice,label='Without variance red')
plt.plot(Start,CallPriceReduced,label='With variance red')
plt.title("Monte-Carlo Heston CallPrice")
plt.legend()
plt.show()

def HestonTheta(S0,Nmc,K,T,r,k,p,N,Theta,Etha,v0,h):
    np.random.seed(0)
    Call,_,_,_,_=HestonCall(S0,Nmc,K,T,r,k,p,N,Theta+h,Etha,v0)
    np.random.seed(0)
    Call2,_,_,_,_=HestonCall(S0,Nmc,K,T,r,k,p,N,Theta-h,Etha,v0)
    GreekTheta=0.5*(Call-Call2)/h
    return GreekTheta
def HestonEtha(S0,Nmc,K,T,r,k,p,N,Theta,Etha,v0,h):
    np.random.seed(0)
    Call,_,_,_,_=HestonCall(S0,Nmc,K,T,r,k,p,N,Theta,Etha+h,v0)
    np.random.seed(0)
    Call2,_,_,_,_=HestonCall(S0,Nmc,K,T,r,k,p,N,Theta,Etha-h,v0)
    GreekEtha=0.5*(Call-Call2)/h
    return GreekEtha
Number=20
Strikes=np.linspace(0,20,Number)
Thetaplot=np.zeros(Number)
Ethaplot=np.zeros(Number)
for i in range(Number):
    Thetaplot[i]=HestonTheta(10,100,Strikes[i],0.5,0.1,3,0.5,100,0.2,0.5,0.04,0.1)
    Ethaplot[i]=HestonEtha(10,100,Strikes[i],0.5,0.1,3,0.5,100,0.2,0.5,0.04,0.1)
plt.plot(Strikes,Thetaplot)
plt.title('GreekTheta')
plt.show()
plt.plot(Strikes,Ethaplot)
plt.title('GreekEtha')
plt.show()
def LevenbergMarquardt(Nmc,N):
    k=3
    epsilon=10**(-4)
    v0=0.04
    p=0.5
    Strike=[8+0.4*i for i in range(21)]
    MarketPrices=[2.0944,1.7488,1.4266,1.1456,0.8919,0.7068,0.5461,0.4187,0.3166,0.2425,0.1860,0.1370,0.0967,0.0715,0.0547,0.0381,0.0306,0.0239,0.0163,0.0139,0.086]
    Theta=0.2
    Etha=0.5
    d=[1,1]
    lamb=0.01
    res=np.zeros(len(Strike))
    J=np.zeros((len(Strike),2))
    HestonPrices=np.zeros(len(Strike))
    while np.linalg.norm(d,2)>epsilon:
        for i in range(len(Strike)):
            Call,_,_,_,_=HestonCall(10,Nmc,Strike[i],0.5,0.01,k,p,N,Theta,Etha,v0)
            HestonPrices[i]=Call
            res[i]=MarketPrices[i]-HestonPrices[i]
            J[i,0]=-HestonTheta(10,Nmc,Strike[i],0.5,0.01,k,p,N,Theta,Etha,v0,0.01)
            J[i,1]=-HestonEtha(10,Nmc,Strike[i],0.5,0.01,k,p,N,Theta,Etha,v0,0.01)
        M=(np.dot(J.T,J)+lamb*np.identity(2))
        d=-np.dot(np.linalg.inv(M),J.T).dot(res.reshape((len(Strike),1)))
        Theta+=d[0]
        Etha+=d[1]
        print(Theta)
        print(Etha)
        print(res)
        print(d)
        if Theta>1 or Theta<0:
            Theta=0.2
        if Etha>1 or Etha<0:
            Etha=0.5
    return Theta,Etha