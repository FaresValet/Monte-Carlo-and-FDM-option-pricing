
# =============================================================================
# We'll be simulating interest rate paths using the Vasicek model
# Interest rates are simulated using an Ornstein-Uhlenbeck Process
# Contrarely to stock prices Interest rates do not rise forever but tend to "return" to some value after a while otherwise it may stunt the economy (if they kept rising)
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
def vasicek(r,a,theta,sigma,T,N,seed=0):
    np.random.seed(seed)  #Generating the same random numbers
    dt=T/N
    Interest_rates=[r]
    for i in range(N):
        dr=(theta-a*Interest_rates[-1])*dt+sigma*np.random.normal()
        Interest_rates.append(Interest_rates[-1]+dr)
    return Interest_rates

x= vasicek(0.05, 0.25, 0.25, 0.02, 30, 200)


def functions(t,T,gamma,etha,sigma):
    B=((1-np.exp(-gamma*(T-t)))/gamma)
    A=((B-(T-t))*(etha*gamma-0.5*sigma**2))/gamma**2 -0.25*(sigma**2 * B**2)/gamma
    return B,A
Time=np.linspace(0.01,200,100)
Y=[0]*100
Y2=[0]*100
Y3=[0]*100

for i in range(100):
    r0=[0.01,0.027,0.05]
    B,A=functions(0,Time[i],0.25,0.25*0.03,0.02)
    Y[i]=-(A-r0[0]*B)/Time[i]
    Y2[i]=-(A-r0[1]*B)/Time[i]
    Y3[i]=-(A-r0[2]*B)/Time[i]

R=[r0[1]]*100 
U=[(0.25*0.03)/0.25 - 0.5*(0.02/0.25)**2]*100
plt.plot(Time,Y,c="r",label='Yield for $r_{0}$=0.01')

plt.plot(Time,Y2,c="b",label='Yield for $r_{0}$=0.027')

plt.plot(Time,Y3,c="y",label='Yield for $r_{0}$=0.05')

plt.title("Yield Curves")
plt.legend()
plt.show()
plt.plot(Time,Y2)
plt.plot(Time,R,c="r",label="r0")
plt.plot(Time,U,c="g",label="ValeurLim")
plt.legend()
plt.title("Yield Curve for r0=0.027")
plt.show()

Time2=np.linspace(0.01,30,100)
Interest=np.linspace(0,1,100)
Ysurf=np.zeros((100,100))
for i in range(100):
    for j in range(100):
        B,A=functions(0,Time[i],0.25,0.25*0.03,0.02)
        Ysurf[i][j]=-(A-Interest[j]*B)/Time[i]
        
fig = plt.figure()
ax = plt.axes(projection='3d')
x,y=np.meshgrid(Time2,Interest)
ax.plot_surface(y,x,Ysurf,cmap="viridis")
ax.view_init(20, 50)
plt.title("Yield Surface")
ax.set_xlabel("Interest rate")

gammar=np.linspace(0.1,0.5,100)
Vol=np.linspace(0,1,100)
Ysurf2=np.zeros((100,100))
for i in range(100):
    for j in range(100):
        B,A=functions(0,1,gammar[i],0.25*0.03,Vol[j])
        Ysurf2[i][j]=-(A-0.5*B)
        
fig = plt.figure()
ax = plt.axes(projection='3d')
x,y=np.meshgrid(gammar,Vol)
ax.plot_surface(y,x,Ysurf2)
ax.view_init(20, 50)
plt.title("Yield Surface 2 ")
ax.set_xlabel("Volatility")

Ethar=np.linspace(0.01,0.04,100)
Ysurf3=np.zeros((100,100))
for i in range(100):
    for j in range(100):
        B,A=functions(0,1,gammar[i],Ethar[j],0.02)
        Ysurf3[i][j]=-(A-0.5*B)
        
fig = plt.figure()
ax = plt.axes(projection='3d')
x,y=np.meshgrid(gammar,Ethar)
ax.plot_surface(y,x,Ysurf3,cmap="coolwarm")
ax.view_init(20, 50)
plt.title("Yield Surface 3")
ax.set_xlabel("Etha")


ax.set_ylabel("Maturity")
plt.show()

ax.set_title('surface')

# =============================================================================
# Part 2 : Calibration of the vasicek model to the Yield curve from the market
# =============================================================================
MarketYields=[0.035,0.041,0.0439,0.046,0.0484,0.0494,0.0507,0.0517,0.052,0.0523]
MarketYields2=[0.056,0.064,0.074,0.081,0.082,0.09,0.087,0.092,0.0895,0.091]
Res=[0]*10
Beta=[0.1,0.1,0.1]
r0=0.023
Epsilon=10**(-9)
lamb=0.01
def B(t,T,gamma):
    return ((1-np.exp(-gamma*(T-t)))/gamma)
def A(t,T,gamma,etha,sigma):
    B=((1-np.exp(-gamma*(T-t)))/gamma)
    return  ((B-(T-t))*(etha*gamma-0.5*sigma**2))/gamma**2 -0.25*(sigma**2 * B**2)/gamma
def Bderivative(gamma,t,T):
    Bder=((T-t)*np.exp(-gamma*(T-t))-B(t,T,gamma))/gamma
    return Bder
def Aderivative(gamma,sigma,etha,t,T):
    B=((1-np.exp(-gamma*(T-t)))/gamma)
    return (etha*(Bderivative(gamma, t, T)*gamma - B)+(T-t)*etha - 0.5*sigma**2 * (Bderivative(gamma, t, T)-2*B/gamma)-((T-t)/gamma) *sigma**2 -0.25*sigma**2 * B * (2*gamma*Bderivative(gamma, t, T) - B))/gamma**2
def YieldCalib(MYields,t,etha,gamma,sigmasquared,epsilon):
    Jacob=np.zeros((10,3))
    Maturity=[3*i for i in range(1,11)]
    d=[1,1,1]
    Yields=[0]*len(MYields)
    while np.linalg.norm(d,2)>epsilon:
            for i in range(len(MYields)):
                
                    Jacob[i][0]=(B(t,Maturity[i],gamma)-(Maturity[i]))/((Maturity[i]-t)*gamma)
            
                    Jacob[i][1]=-((B(t,Maturity[i],gamma)-Maturity[i]+t)/(2*gamma) + 0.25*B(t,Maturity[i],gamma)**2)*1/((Maturity[i]-t)*gamma)
                
                    Jacob[i][2]=1/(Maturity[i]-t) * (Aderivative(gamma, np.sqrt(sigmasquared), etha, t, Maturity[i]) - r0*Bderivative(gamma,t, Maturity[i]))
        
        
                    Yields[i]=-(A(t,Maturity[i],gamma,etha,np.sqrt(sigmasquared))-r0*B(t,Maturity[i],gamma))/(Maturity[i]-t)
                    Res[i]=MYields[i]-Yields[i]
        
        
            d=-np.dot(np.linalg.inv(np.dot(Jacob.T,Jacob)+lamb*np.identity(3)),np.dot(Jacob.T,Res))
            etha+=d[0]
            sigmasquared+=d[1]
            gamma+=d[2]
    print("Parameters for t=0 are", [etha,sigmasquared,gamma])
    
    plt.scatter(Maturity,MYields,c='r',label='Market Yields')
    plt.plot(Maturity,Yields,c='b',label='Vasicek Yields')
    plt.title('Market Yields and Theoretical Yields with Calibrated Model at t=%d' %(t))
    plt.legend()
    plt.show()

# =============================================================================
# Part 3: Calibration to historical dates and linear regression to find the best predictive linear pattern of interest rates
# =============================================================================

def interestrate(r,etha,gamma,sigma,T,step,a,b):
    lamb=0.01
    dt=T/step
    Rate=[0]*step
    Rate[0]=r
    for i in range(step-1):
        Rate[i+1]=Rate[i]*np.exp(-gamma*dt) + etha/gamma * (1-np.exp(-gamma*dt)) + np.sqrt(sigma**2 * (1-np.exp(-2*gamma*dt))/2*gamma)*np.random.normal()
    
    Jacob=np.zeros((step,2))
    d=[0.1,0.1]
    Res=[0]*step
    Epsilon=10**(-9)
    count=0
    a_theory = np.exp(-gamma * dt)
    b_theory = etha / gamma * (1 - np.exp(-gamma * dt))
    while np.linalg.norm(d)>Epsilon:
       
            for i in range(step-1):
                
               Jacob[i][0]=-Rate[i]
                
               Jacob[i][1]=-1
                
               Res[i]=Rate[i+1]-a*Rate[i]-b
        
        
            d=-np.dot(np.linalg.inv(np.dot(Jacob.T,Jacob)+lamb*np.identity(2)),np.dot(Jacob.T,Res))
            a+=d[0]
            b+=d[1]
            
            count=count+1
        
    print('Our parameters a and b are ', [a,b])
    yapproach=[i*a + b for i in Rate[:step-1]]
    Error=[a_i - b_i for a_i, b_i in zip(Rate[1:step], yapproach)]
    Var=0
    for i in range(step-1):
        Var+=(Rate[i+1]-a*Rate[i]-b)**2
    
    Variance=Var/step
    plt.scatter(Rate[:step-1],Rate[1:step], label="interest rate data")
    plt.plot(Rate[:step-1],yapproach,c='g', label="linear approximation")
    plt.plot(Rate[:step-1],a_theory*np.array(Rate[:step-1])+b_theory,c='r', label="True line")
    plt.title("Linear regression to find the line which fits the interest rate prediction the best y(r_i)=r_(i+1)")
    plt.legend()
    plt.show()
    print("Variance is equal to ", np.var(Error))
    """x=x[:step]
    x=np.array(x)
    y=np.array(y)
    x=x.reshape(x.shape[0],1)
    y=y.reshape(y.shape[0],1)
    X=np.hstack((x,np.ones(x.shape)))
    theta=np.linalg.inv(np.dot(X.T,X)).dot(np.dot(X.T,y))"""
    TheoreticalGamma=-np.log(a)/dt
    TheoreticalEtha=TheoreticalGamma*(b)/(1-a)
    TheoreticalSigma=np.std(Error)*np.sqrt(-2*np.log(a)/(dt*(1-a**2)))
    TheoreticalVector=[TheoreticalEtha,TheoreticalGamma,TheoreticalSigma]
    RealGamma=-np.log(a_theory)/dt
    RealEtha=RealGamma*(b_theory)/(1-a_theory)
    RealSigma=np.sqrt(Variance)*np.sqrt(-2*np.log(a_theory)/(dt*(1-a_theory**2)))
    RealVector=[RealEtha,RealGamma,RealSigma]
    return TheoreticalVector,RealVector,Jacob