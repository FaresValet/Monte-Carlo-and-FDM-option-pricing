
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
Maturity=[3*i for i in range(1,11)]
MarketYields=[0.035,0.041,0.0439,0.046,0.0484,0.0494,0.0507,0.0517,0.052,0.0523]
Yields=[0]*10
Res=[0]*10
d=np.array([1]*3)
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

Jacob=np.zeros((10,3))
while np.linalg.norm(d,2)>Epsilon:
    for j in range(3):
        for i in range(10):
            if j==0:
                Jacob[i][j]=(B(0,Maturity[i],Beta[2])-(Maturity[i]))/(Maturity[i]*Beta[2])
            elif j==1:
                Jacob[i][j]=-((B(0,Maturity[i],Beta[2])-Maturity[i])/(2*Beta[2]) + 0.25*B(0,Maturity[i],Beta[2])**2)*1/(Maturity[i]*Beta[2])
            else:
                Jacob[i][j]=1/Maturity[i] * (Aderivative(Beta[2], np.sqrt(Beta[1]), Beta[0], 0, Maturity[i]) - r0*Bderivative(Beta[2],0, Maturity[i]))
    
    for i in range(10):
        Yields[i]=-(A(0,Maturity[i],Beta[2],Beta[0],np.sqrt(Beta[1]))-r0*B(0,Maturity[i],Beta[2]))/Maturity[i]
        Res[i]=MarketYields[i]-Yields[i]
    
    for j in range(3):
        d=-np.dot(np.linalg.inv(np.dot(Jacob.T,Jacob)+lamb*np.identity(3)),np.dot(Jacob.T,Res))
        Beta[j]=Beta[j]+d[j]
print("Parameters for t=0 are", Beta)

plt.scatter(Maturity,MarketYields,c='r',label='Market Yields')
plt.plot(Maturity,Yields,c='b',label='Vasicek Yields')
plt.title("Market Yields and Theoretical Yields with Calibrated Model at t=0")
plt.legend()
plt.show()
# =============================================================================
# Part 3 : Recalibration of the yield curve with t=1 (one year) data
# =============================================================================
Beta2=[0.05,0.05,0.05]
Jacob2=np.zeros((10,3))
Res2=[0]*10
d2=np.array([1]*3)
MarketYields2=[0.056,0.064,0.074,0.081,0.082,0.09,0.087,0.092,0.0895,0.091]
Yields2=[0]*10
while np.linalg.norm(d2,2)>Epsilon:
    for j in range(3):
        for i in range(10):
            if j==0:
                Jacob2[i][j]=(B(1,Maturity[i],Beta2[2])-(Maturity[i])+1)/((Maturity[i]-1)*Beta2[2])
            elif j==1:
                Jacob2[i][j]=-((B(1,Maturity[i],Beta2[2])-Maturity[i]+1)/(2*Beta2[2]) + 0.25*B(1,Maturity[i],Beta2[2])**2)*1/((Maturity[i]-1)*Beta2[2])
            else:
                Jacob2[i][j]=1/(Maturity[i]-1) * (Aderivative(Beta2[2], np.sqrt(Beta2[1]), Beta2[0], 1, Maturity[i]) - r0*Bderivative(Beta2[2],1, Maturity[i]))
    
    for i in range(10):
        Yields2[i]=-(A(1,Maturity[i],Beta2[2],Beta2[0],np.sqrt(Beta2[1]))-r0*B(1,Maturity[i],Beta2[2]))/(Maturity[i]-1)
        Res2[i]=MarketYields2[i]-Yields2[i]
    
    for j in range(3):
        d2=-np.dot(np.linalg.inv(np.dot(Jacob2.T,Jacob2)+lamb*np.identity(3)),np.dot(Jacob2.T,Res2))
        Beta2[j]=Beta2[j]+d2[j]
print("Parameters for t=1 are", Beta2)

plt.scatter(Maturity,MarketYields2,c='r',label='Market Yields')
plt.plot(Maturity,Yields2,c='b',label='Vasicek Yields')
plt.title("Market Yields and Theoretical Yields with Calibrated Model at t=1")
plt.legend()
plt.show()
# =============================================================================
# Part 4: Calibration to historical dates and linear regression to find the best predictive linear pattern of interest rates
# =============================================================================
Vect=[1,1]
step=100
def interestrate(r,etha,gamma,sigma,T,N):
    dt=T/N
    Rate=[r]
    for i in range(N):
        Rate.append(Rate[-1]*np.exp(-gamma*dt) + etha/gamma * (1-np.exp(-gamma*dt)) + np.sqrt(sigma**2 * (1-np.exp(-2*gamma*dt))/2*gamma)*np.random.normal(0,1))
    
    return Rate
x=interestrate(0.10,0.6,4,0.08,5,step)
y=x[1:]
plt.plot(x)
plt.title(" theoretical interest rate")
plt.show()
Jacob3=np.zeros((step,2))
Data1=x[:step]
d3=[0.1,0.1]
Droite=[0]*step
Res3=[0]*step
Epsilon=10**(-9)
while np.linalg.norm(d3,2)>Epsilon:
    for j in range(2):
        for i in range(step):
            if j==0:
                Jacob3[i][j]=-Data1[i]
            elif j==1:
                Jacob3[i][j]=-1
            
    
    for i in range(step):
        Droite[i]=Vect[0]*Data1[i]+Vect[1]
        Res3[i]=y[i]-Droite[i]
    
    for j in range(2):
        d3=-np.dot(np.linalg.inv(np.dot(Jacob3.T,Jacob3)+lamb*np.identity(2)),np.dot(Jacob3.T,Res3))
        Vect[j]=Vect[j]+d3[j]
print('Our parameters a and b are ', Vect)
yapproach=[i*Vect[0]+Vect[1] for i in Data1]
Error=[a_i - b_i for a_i, b_i in zip(y, yapproach)]
plt.scatter(x[:step],y, label="interest rate data")
plt.plot(x[:step],yapproach,c='g', label="linear approximation")
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
dt=5/step
TheoreticalGamma=-np.log(Vect[0])/dt
TheoreticalEtha=TheoreticalGamma*(Vect[1])/(1-Vect[0])
TheoreticalSigma=np.std(Error)*np.sqrt(-2*np.log(Vect[0])/(dt*(1-Vect[0]**2)))
TheoreticalVector=[TheoreticalEtha,TheoreticalGamma,TheoreticalSigma]