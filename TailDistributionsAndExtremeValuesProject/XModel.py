import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import root_scalar
import math
from datetime import datetime
from pyextremes import plot_mean_residual_life
from scipy.stats import kstest
import copulae
import random
from statsmodels.distributions.copula.api import (
    CopulaDistribution, GumbelCopula, IndependenceCopula)
# =============================================================================
# Importing our data
# =============================================================================
df=pd.read_excel('h:\Downloads\Données.xlsx')
data=df.loc[:,'X']

datay=df[["date", "Y"]].set_index("date").resample("Y").max()
data1 = df[["date", "X"]]
data1 = data1.set_index("date")
data1=data1.resample("Y").max() #Block maxima (by year) for generalized extreme value fitting
# =============================================================================
# Mean excess function to estimate the threshold for generalized pareto fitting, we estimate the function empirically and we compare it with the pyextremes library function
# =============================================================================
def Excess(u):
    count=0
    Max=np.zeros(len(data))
    for i in range(len(data)):
        Max[i]=np.maximum(data[i]-u,0)
        if data[i]>u:
            count+=1
    return Max.sum()/count
ParetoSample=[1000*i for i in range(1,35)]
ExcessPlot=[0]*len(ParetoSample)
for i in range(len(ParetoSample)):
    ExcessPlot[i]=Excess(ParetoSample[i])

plt.plot(ParetoSample,ExcessPlot,c="r",label="e(x)")
plt.title("Mean Excess Plot")
plt.legend()
plt.show() #We will take 20000 as a threshold based on the mean excess plot for now
plot_mean_residual_life(data)
plt.title("Mean Excess Plot using Pyextremes Library")
plt.show()
# =============================================================================
# Start of simulation
# =============================================================================
datapar=[x for x in data if x>=16000] #Around 1 years data-clustering
v=np.ceil(np.log2(len(data1))) + 1 #This is called Sturge's rule, this formula is used to calculate the adequate number of bins to visualize your data's distribution
v2=np.ceil(np.log2(len(data))) + 1
v3=np.ceil(np.log2(len(datapar))) + 1
y,x=np.histogram(data1,bins=int(v),density=True) #"clustering" our data for the plot
y2,x2=np.histogram(data,bins=int(v2),density=True)
y3,x3=np.histogram(datapar,bins=int(v3),density=True)
plt.hist(data, bins=int(v2), density=True)

plt.title("Histogram")
plt.show()
x = (x + np.roll(x, -1))[:-1] / 2.0
x2 = (x2 + np.roll(x2, -1))[:-1] / 2.0 
x3 = (x3 + np.roll(x3, -1))[:-1] / 2.0 #This takes the mid point of every "bin" interval as the reference x-axis point for its corresponding y probability
# =============================================================================
# Fitting our data and plotting the PDFs
# =============================================================================
fit1=stats.genextreme.fit(data1,loc=0) #The fit method finds the optimal parameters for your data fitting a chosen probability distribution
fit2=stats.norm.fit(data)
fit3=stats.genpareto.fit(datapar,loc=0)
fitY=stats.genpareto.fit(datay,loc=0)
pdf=stats.genextreme.pdf(x,*fit1)
pdf2=stats.norm.pdf(x2,*fit2)
pdf3=stats.genpareto.pdf(x3,*fit3)
plt.plot(x2,pdf2,c='r',label='Gaussian Law')
plt.plot(x2,y2,c='g',label='Real Data ')
plt.legend()
plt.title(" Data and Gaussian densities")
plt.show()
plt.plot(x,pdf,c='b',label='GEV')
plt.plot(x,y,c='g',label='Real Data ')
plt.legend()
plt.title("Data and GEV probability densities")
plt.show()
plt.plot(x3,pdf3,c='y',label='GPD')
plt.plot(x3,y3,c='g',label='Real Data ')
plt.legend()
plt.title(" Data and GPD probability densities")
plt.show() 
#WARNING: The graphics ARE NOT PROPERLY SCALED AND  do not tell the whole story because they are not normalized, a way to find the best fitting model is to plot the theoretical cumulative distribution of the tail  and the empirical one from our data, to get an accurate depiction of what's actually happening check the next PLOTS
# =============================================================================
# Finding the best fit using least-squares/kolmogorov-smirnoff
# =============================================================================
#Since we are studying the tail's behaviour we can study  P(X>u)=1-P(X<u) for u sufficiently large
#Empirical distribution 
def empirical(x,u):
    count=0
    for i in range(len(x)):
        if x[i]>u:
            count+=1
    
    return count/len(x)
#In order to compare with our Gaussian/GEV/GPD distributions we will simulate values from those and compare with our empirical distribution
Xaxis=np.linspace(20000,90000,50)
YaxisEmpiricalDistribution=np.zeros(len(Xaxis))  
YaxisEmpiricalDistributionYearly=np.zeros(len(Xaxis)) 
YaxisEmpiricalDistribution2Years=np.zeros(len(Xaxis)) 
YaxisGaussian=np.zeros(len(Xaxis)) 
YaxisGev=np.zeros(len(Xaxis)) 
YaxisGPD=np.zeros(len(Xaxis))   
dataGaussian=stats.norm.rvs(*fit2,size=len(data)) #we generate len(data) number of values from our distributions and plot their respective probabilities
dataGPD=stats.genpareto.rvs(*fit3,1000)
dataGev=stats.genextreme.rvs(*fit1,1000) 
for i in range(len(Xaxis)):
    YaxisEmpiricalDistribution[i]=empirical(data,Xaxis[i])
    YaxisEmpiricalDistributionYearly[i]=empirical(np.array(data1),Xaxis[i])
    YaxisEmpiricalDistribution2Years[i]=empirical(datapar,Xaxis[i])
    YaxisGaussian[i]=empirical(dataGaussian,Xaxis[i])
    YaxisGev[i]=empirical(dataGev,Xaxis[i])
    YaxisGPD[i]=empirical(dataGPD,Xaxis[i])

plt.plot(Xaxis,YaxisEmpiricalDistribution,c='g',label='Original Data Disitribution')
plt.legend()
plt.title("Empirical distribution P(X>u)=1-P(X<u) ")
plt.show()
plt.plot(Xaxis,YaxisEmpiricalDistribution,c='g',label='Original Data Sample')
plt.plot(Xaxis,YaxisGaussian,c='b',label='Scaled Gaussian Sample')
plt.legend()
plt.title("Empirical distribution P(X>u)=1-P(X<u) Gaussian and original  ")
plt.show()
plt.plot(Xaxis,YaxisEmpiricalDistribution2Years,c='g',label='Original Data Sample')
plt.plot(Xaxis,YaxisGPD,c='r',label='GPD Data Sample')
plt.title("Empirical distribution P(X>u)=1-P(X<u) GPD and original ")
plt.legend()
plt.show()
plt.plot(Xaxis,YaxisEmpiricalDistributionYearly,c='g',label='Original Data Sample')
plt.plot(Xaxis,YaxisGev,c='y',label='GEV Data Sample')
plt.title("Empirical distribution P(X>u)=1-P(X<u) GEV  and original")
plt.legend()
plt.show()
#Least-squares
def leastsquares(EmpiricalDist,Dist): #Function to find the best fitting model
    Error=0
    for i in range(len(Xaxis)):
        Error+=(EmpiricalDist[i]-Dist[i])**2
    return Error/len(Xaxis)

#quantiles
def Quantile(alpha,data):
    Sortedata=np.sort(data)
    IndexData=math.floor(len(data)*alpha)
    return Sortedata[IndexData]
#Kolmogorov-Smirnoff
x=kstest(data,stats.norm.cdf,args=fit2)
x3=kstest(data1,stats.genextreme.cdf,args=fit1)
x2=kstest(datapar,stats.genpareto.cdf,args=fit3)
print("Kstest for gaussian",x) #We reject the null hypothesis since the p value is this small, our data does not follow a normal distribution
print("Kstest for GPD",x2)  
print("Kstest for GEV",x3)  
def BIC(x, x_theo, parmater):
    n = len(x)
    residual = np.subtract(x_theo, x)
    SSE = np.sum(np.power(residual, 2))
    return n * np.log(SSE / n) + parmater * np.log(n)


def AIC(x, x_theo, parmater):
    n = len(x)
    residual = np.subtract(x_theo, x)
    rss = np.sum(np.power(residual, 2))
    return n * np.log(rss / n) + 2 * parmater

#Experimental code
ReturnPeriods=[12*i for i in range(2,10)]
Alphas=[1-1/x for x in ReturnPeriods]
Alphas2=[1-12/x for x in ReturnPeriods]
Alphas3=[1-12/x for x in ReturnPeriods]

Quantiledata=np.zeros(len(Alphas))
QuantileGEV=np.zeros(len(Alphas))
QuantileGPD =np.zeros(len(Alphas))
QuantileGauss=np.zeros(len(Alphas))

for i in range(len(Alphas)):
    Quantiledata[i]=Quantile(Alphas[i],data)
    QuantileGauss[i]=Quantile(Alphas[i],stats.norm.rvs(*fit2,100000))
    QuantileGEV[i]=Quantile(Alphas2[i],stats.genextreme.rvs(*fit1,100000))
    QuantileGPD[i]=Quantile(Alphas3[i],stats.genpareto.rvs(*fit3,100000))
    
plt.scatter(Quantiledata,QuantileGauss)
plt.plot(Quantiledata,Quantiledata)
plt.title("Q-Q plot Gauss and Original Data")
plt.show()
plt.scatter(Quantiledata,QuantileGEV)
plt.plot(Quantiledata,Quantiledata)
plt.title("Q-Q plot GEV Original Data")
plt.show()
plt.scatter(Quantiledata,QuantileGPD)
plt.plot(Quantiledata,Quantiledata)
plt.title("Q-Q plot GPD Original Data")
plt.show()
#We select GPD based on the p-value and the Q-Q plot.
# =============================================================================
#  Quantiles 
# =============================================================================
#Actual Data
SortedData=np.sort(df.loc[:,'X'])
alpha=0.995
alpha2=1-1/(200*12)
alphaquantile=1-(12/200)
IndexData=math.floor(len(df.loc[:,'X'])*alpha)
QuantileData=SortedData[IndexData]
print("Quantile empirique est", QuantileData)
#Gaussian law
SortedGaussian=np.sort(stats.norm.rvs(*fit2,size=1000000))
IndexGauss=math.floor(1000000*alpha2)
IndexQuantileGauss=math.floor(1000000*alpha)
QuantileGaussian99=SortedGaussian[IndexQuantileGauss]
QuantileGaussian=SortedGaussian[IndexGauss]
print("Quantile 99.5 de la loi normale  est", QuantileGaussian99)
#GPD 
SortedGPD=np.sort(stats.genpareto.rvs(*fit3,size=100000))
IndexGPD=math.floor(100000*alpha)
ReturnPareto=SortedGPD[IndexGPD]
IndexQuantileGPD=math.floor(100000*alphaquantile)
QuantilePareto=SortedGPD[IndexQuantileGPD]
print("Quantile 99.5 GPD est", QuantilePareto)
#GEV
SortedGEV=np.sort(stats.genextreme.rvs(*fit1,size=100000))
IndexGEV=math.floor(100000*alpha)
IndexQuantileGEV=math.floor(100000*alphaquantile)
ReturnGEV=SortedGEV[IndexGEV]
QuantileGEV=SortedGEV[IndexQuantileGEV]
print("Quantile 99.5 GEV est", QuantileGEV)
QuantileVector=[QuantileData,QuantileGEV,QuantilePareto]
print("Model X 99.5 Quantiles (Monthly)  values for OriginalData,GEV, and GPD are", QuantileVector)
ReturnVector=[QuantileGaussian,ReturnGEV,ReturnPareto]
print("Model X 200 years return period  values for Gaussian,GEV, and GPD are", ReturnVector)
# =============================================================================
# X and Y Model dependency :
# =============================================================================
data = pd.read_excel("data\Base X Y.xlsx")
X = data.iloc[:, 1]
Y = data.iloc[:, 2]
data3 = data[["X", "Y"]]

_, ndim = data3.shape
copulaGaussianCopula = copulae.GaussianCopula(dim=ndim)
paramGaussianCopula = copulaGaussianCopula.fit(data3)

copulaFrankCopula = copulae.FrankCopula(dim=ndim)
paramFrankCopula = copulaFrankCopula.fit(data3)

copulaClaytonCopula = copulae.ClaytonCopula(dim=ndim)
paramClaytonCopula = copulaClaytonCopula.fit(data3)

copulaStudentCopula = copulae.StudentCopula(dim=ndim)
paramStudentCopula = copulaStudentCopula.fit(data3)

copulaGumbelCopula = copulae.GumbelCopula(dim=ndim)
paramGumbelCopula = copulaGumbelCopula.fit(data3)

theta_Frank = paramFrankCopula.params
theta_Gumbel = paramGumbelCopula.params
theta_Clayton = paramClaytonCopula.params
Degree_Freedom_student = paramStudentCopula.params[0]
corr_student = paramStudentCopula.params[1][0]
corr_matrix = paramGaussianCopula.params[0]

print(" theta_Frank =", theta_Frank)
print(" theta_Gumbel =", theta_Gumbel)
print(" Degree_Freedom =", corr_student)
print(" theta_Clayton =", theta_Clayton)
print(" corr_matrix =", corr_matrix)

means = [0.500000, 0.500000]
a = 0.288434 ** 2
b = 0.288434 * 0.288434 * corr_matrix
cov_matrix = [[a, b], [b, a]]
print(cov_matrix)
copula = GumbelCopula(theta=theta_Gumbel)
copula.plot_scatter()
plt.title("Gumbel Copula "+str(theta_Gumbel))
aic = AIC(np.array(df.values), np.array(copula.rvs(nobs=598, random_state=None)), 2)
print("AIC = ", aic)
bic = BIC(np.array(df.values), np.array(copula.rvs(nobs=598, random_state=None)), 2)
print("BIC = ", bic)
ks = Kolmogorov_Smirnov(np.array(df.values)[:,0], np.array(copula.rvs(nobs=598, random_state=None))[:,0])
print("Kolmogorov_Smirnov = ", ks)
plt.annotate("AIC = " + str(aic), xy=(0, 1))
plt.annotate("BIC = " + str(bic), xy=(0, 0.95))
plt.annotate("Ks = " + str(ks), xy=(0, 0.9))
plt.savefig("Graph/Gumbel_Copula.png")
plt.show()

copula = ClaytonCopula(theta=theta_Clayton)
copula.plot_scatter()
plt.title("Clayton Copula " + str(theta_Clayton))
aic = AIC(np.array(df.values), np.array(copula.rvs(nobs=598, random_state=None)), 2)
print("AIC = ", aic)
bic = BIC(np.array(df.values), np.array(copula.rvs(nobs=598, random_state=None)), 2)
print("BIC = ", bic)
ks = Kolmogorov_Smirnov(np.array(df.values)[:,0], np.array(copula.rvs(nobs=598, random_state=None))[:,0])
print("Kolmogorov_Smirnov = ", ks)
plt.annotate("AIC = " + str(aic), xy=(0, 1))
plt.annotate("BIC = " + str(bic), xy=(0, 0.95))
plt.annotate("Ks = " + str(ks), xy=(0, 0.9))
plt.savefig("Graph/Clayton_Copula.png")
plt.show()

copula = FrankCopula(theta=theta_Frank)
copula.plot_scatter()
plt.title("Frank Copula " + str(theta_Frank))
aic = AIC(np.array(df.values), np.array(copula.rvs(nobs=598, random_state=None)), 2)
print("AIC = ", aic)
bic = BIC(np.array(df.values), np.array(copula.rvs(nobs=598, random_state=None)), 2)
print("BIC = ", bic)
ks = Kolmogorov_Smirnov(np.array(df.values)[:,0], np.array(copula.rvs(nobs=598, random_state=None))[:,0])
print("Kolmogorov_Smirnov = ", ks)
plt.annotate("AIC = " + str(aic), xy=(0, 1))
plt.annotate("BIC = " + str(bic), xy=(0, 0.95))
plt.annotate("Ks = " + str(ks), xy=(0, 0.9))
plt.savefig("Graph/Frank_Copula.png")
plt.show()

copula = StudentTCopula(df=Degree_Freedom_student , corr=corr_student)
copula.plot_scatter()
plt.title("Student Copula " + str(corr_student) +" df = "+str(Degree_Freedom_student))
# print(np.array(copula.rvs(nobs=598, random_state=None)).shape)
aic = AIC(np.array(df.values), np.array(copula.rvs(nobs=598, random_state=None)), 2)
print("AIC = ", aic)
bic = BIC(np.array(df.values), np.array(copula.rvs(nobs=598, random_state=None)), 2)
print("BIC = ", bic)
ks = Kolmogorov_Smirnov(np.array(df.values)[:,0], np.array(copula.rvs(nobs=598, random_state=None))[:,0])
print("Kolmogorov_Smirnov = ", ks)
plt.annotate("AIC = "+str(aic),xy=(0,1))
plt.annotate("BIC = "+str(bic),xy=(0,0.95))
plt.annotate("Ks = "+str(ks),xy=(0,0.9))
plt.savefig("Graph/Student_Copula.png")
plt.show()

copula = GaussianCopula(corr=corr_matrix)
copula.plot_scatter()
plt.title("Gaussian Copula " + str(corr_matrix))
aic = AIC(np.array(df.values), np.array(copula.rvs(nobs=598, random_state=None)), 2)
print("AIC = ", aic)
bic = BIC(np.array(df.values), np.array(copula.rvs(nobs=598, random_state=None)), 2)
print("BIC = ", bic)
ks = Kolmogorov_Smirnov(np.array(df.values)[:,0], np.array(copula.rvs(nobs=598, random_state=None))[:,0])
print("Kolmogorov_Smirnov = ", ks)
plt.annotate("AIC = " + str(aic), xy=(0, 1))
plt.annotate("BIC = " + str(bic), xy=(0, 0.95))
plt.annotate("Ks = " + str(ks), xy=(0, 0.9))
plt.savefig("Graph/Gaussian_Copula.png")
plt.show()


Obs = copulae.pseudo_obs(df.set_index("date")) #Observations pour générer le nuage de copules
emp_cop = copulae.EmpiricalCopula(Obs, smoothing="beta")
data = emp_cop.data
plt.scatter(data['X'], data['Y'])
plt.xlabel(" X")
plt.ylabel(" Y")
plt.title("Copule de X et Y") #rangs de Y en fonction des rangs de X
plt.legend()
plt.show()

Obs = copulae.pseudo_obs(df.set_index("date").resample("Y").max())
_,ndim=data.shape
C_cop=copulae.ClaytonCopula(dim=ndim)  #Je definis les copules par rapport auxquelles on va tester nos observations
G_cop=copulae.GaussianCopula(dim=ndim)
F_cop=copulae.FrankCopula(dim=ndim)
K_cop=copulae.GumbelCopula(dim=ndim)
T_cop=copulae.StudentCopula(dim=ndim)
x1=C_cop.fit(Obs) #Ici on "fit" nos copules aux observations J'utilise ensuite dans la console la methode log_lik du module copulae afin de trouver celle avec le maximum de vraisemblance le plus élevé et on trouve Gumbel pour un paramètre thêta=2,9 ..
x2=G_cop.fit(Obs)
x3=F_cop.fit(Obs)
x4=K_cop.fit(Obs)
x5=T_cop.fit(Obs)
XSample=stats.genpareto.rvs(*fit3,size=1) #yearly max
u=stats.genpareto.cdf(XSample,*fit3)
t1=random.uniform(0, 1)
t2=random.uniform(0, 1)
theta=2.966099153905645
def GCopula(x): #ne sert à rien, j'explorais juste un autre moyen afin de trouver le retour combiné
    XSample=stats.genpareto.rvs(*fit3,size=1) #yearly max
    u=stats.genpareto.cdf(XSample,*fit3)
    t1=random.uniform(0, 1)
    theta=2.966099153905645
    return np.exp(-((-np.log(u))**(theta) +(-np.log(x))**(theta))**(1/theta))-t1
def Generator(t):
    return (-np.log(t))**theta
def DerivativeGenerator(t):
    return (theta*(-np.log(t))**theta)/(t*np.log(t))
def K(t):
    return (t-Generator(t)/DerivativeGenerator(t))-t2


copula = GumbelCopula(theta=2.966099153905645) #copule que l'on retient pour simuler la dépendance X et Y
_ = copula.plot_pdf()  # returns a matplotlib figure
plt.title('Gumbel Copula for theta=2.966')
plt.show()
"""Sample=copula.rvs(10000) #Génère 10000 couples de notre copule
QuantileX=np.zeros(10000)
QuantileY=np.zeros(10000)
fitY=(-0.2115559548242755, 3819.72758293294, 415.2103277015068) #◙Paramètres de Y (trouvée dans le code de Y)
for i in range(10000):
    QuantileX[i]=np.quantile(stats.genpareto.rvs(*fit3,size=100000),Sample[i,0])
    QuantileY[i]=np.quantile(stats.genpareto.rvs(*fitY,size=100000),Sample[i,1])
#La boucle en haut génère des réalisations X,Y qui tiennent compte de la structure de dépendance
XYCombined=np.add(QuantileX,QuantileY) #Maintenant que l'on tient compte de la structure de dépendance on peut ajouter les quantiles
Quantile=np.quantile(stats.genpareto.rvs(*fit3,size=100000),0.995)
print("Combined 200 years return is", np.quantile(XYCombined,0.995))"""
