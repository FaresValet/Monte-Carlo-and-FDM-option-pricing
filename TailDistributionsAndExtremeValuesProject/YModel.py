import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import stats
from pyextremes import plot_mean_residual_life
import math
from scipy.stats import kstest
# =============================================================================
#Importing files
# =============================================================================
df=pd.read_excel('h:\Downloads\Données.xlsx')
data=df.loc[:,'Y']
data1= df[["date", "Y"]]
data1= data1.set_index("date")
data1=data1.resample("Y").max()
# =============================================================================
# Pareto threshold selection and histograms
# =============================================================================

plot_mean_residual_life(data)
plt.title("Mean Excess Plot using Pyextremes Library")
plt.show()
datapar=[x for x in df.loc[:,'Y']  if x>=3800]
v=np.ceil(np.log2(len(data))) + 1
v2=np.ceil(np.log2(len(data1))) + 1 
v3=np.ceil(np.log2(len(datapar))) + 1 #This is called Sturge's rule, this formula is used to calculate the adequate number of bins to visualize your data's distribution
y,x=np.histogram(data,bins=int(v),density=True)
y2,x2=np.histogram(data1,bins=int(v2),density=True)
y3,x3=np.histogram(datapar,bins=int(v3),density=True) #"clustering" our data for the plot
x = (x + np.roll(x, -1))[:-1] / 2.0 
x2 = (x2 + np.roll(x2, -1))[:-1] / 2.0
x3 = (x3 + np.roll(x3, -1))[:-1] / 2.0#This takes the mid point of every "bin" interval as the reference x-axis point for its corresponding y probability
# =============================================================================
# Fitting our data and plotting the PDFs
# =============================================================================
fit1=stats.genextreme.fit(data1,loc=8) #The fit method finds the optimal parameters for your data fitting a chosen probability distribution
fit2=stats.norm.fit(data)
fit3=stats.genpareto.fit(datapar,loc=8)
pdf=stats.genextreme.pdf(x2,*fit1)
pdf2=stats.norm.pdf(x,*fit2)
pdf3=stats.genpareto.pdf(x3,*fit3)
plt.plot(x,pdf2,c='r',label='Gaussian Law')
plt.plot(x,y,c='g',label='Real Data ')
plt.legend()
plt.title(" Data and Gaussian densities")
plt.xlim(3400,5000)
plt.ylim(0,0.001)
plt.show()
plt.plot(x2,pdf,c='b',label='GEV')
plt.plot(x2,y2,c='g',label='Real Data ')
plt.legend()
plt.title("Data and GEV probability densities")
plt.show()
plt.plot(x3,pdf3,c='y',label='GPD')
plt.plot(x3,y3,c='g',label='Real Data ')
plt.legend()
plt.title(" Data and GPD probability densities")
plt.show()

# =============================================================================
# Finding the best fit using least-squares
# =============================================================================
def empirical(x,u):
    count=0
    for i in range(len(x)):
        if x[i]>u:
            count+=1
    
    return count/len(x)
#In order to compare with our Gaussian/GEV/GPD distributions we will simulate values from those and compare with our empirical distribution
Xaxis=np.linspace(3000,6000,50)
YaxisEmpiricalDistribution=np.zeros(len(Xaxis))  
YaxisGaussian=np.zeros(len(Xaxis)) 
YaxisGev=np.zeros(len(Xaxis)) 
YaxisGPD=np.zeros(len(Xaxis))   
dataGaussian=stats.norm.rvs(*fit2,size=len(data)) #we generate len(data) number of values from our distributions and plot their respective probabilities
dataGPD=stats.genpareto.rvs(*fit3,size=1000)
dataGev=stats.genextreme.rvs(*fit1,size=1000) 
YaxisEmpiricalDistributionYear=np.zeros(len(Xaxis))
for i in range(len(Xaxis)):
    YaxisEmpiricalDistribution[i]=empirical(data,Xaxis[i])
    YaxisEmpiricalDistributionYear[i]=empirical(np.array(data1),Xaxis[i])
    YaxisEmpiricalDistributionYear[i]=empirical(datapar,Xaxis[i])
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
plt.title("Empirical distribution P(X>u)=1-P(X<u) Gaussian and Original Data ")
plt.show()
plt.plot(Xaxis,YaxisEmpiricalDistributionYear,c='g',label='Original Data Sample')
plt.plot(Xaxis,YaxisGPD,c='r',label='GPD Data Sample')
plt.title("Empirical distribution P(X>u)=1-P(X<u) GPD and Original Data")
plt.legend()
plt.show()
plt.plot(Xaxis,YaxisEmpiricalDistributionYear,c='g',label='Original Data Sample')
plt.plot(Xaxis,YaxisGev,c='y',label='GEV Data Sample')
plt.title("Empirical distribution P(X>u)=1-P(X<u) GEV and Original Data")
plt.legend()
plt.show()
plt.plot(Xaxis,YaxisEmpiricalDistribution,c='g',label='Original Data Sample')
plt.plot(Xaxis,YaxisGaussian,c='b',label='Gaussian')
plt.plot(Xaxis,YaxisGPD,c='r',label='GPD')
plt.plot(Xaxis,YaxisGev,c='y',label='GEV')
plt.title("All distributions plotted together")
plt.legend()
plt.show()
#Least-squares
def leastsquares(Dist): #Function to find the best fitting model
    Error=0
    for i in range(len(Xaxis)):
        Error+=(YaxisEmpiricalDistribution[i]-Dist[i])**2
    return Error/len(Xaxis)
#Kolmogorov-Smirnoff
x=kstest(data,stats.norm.cdf,args=fit2)
x2=kstest(datapar,stats.genpareto.cdf,args=fit3)
print("Kstest for gaussian",x) #P value is pretty high we cannot reject the gaussian distribution, also from the p value from the GPD, the tail seems to be more gaussian than anything else
print("Kstest for GPD",x2)  

#We select the Gaussian model
# =============================================================================
#  Quantiles 
# =============================================================================
#Actual Data
SortedData=np.sort(data1)
alphapareto=0.995
alpha=0.995
alpha2=0.9995
IndexData=math.floor(len(data1)*alpha)
QuantileData=SortedData[IndexData]
#Gaussian law
SortedGaussian=np.sort(stats.norm.rvs(*fit2,size=1000000))
IndexGauss=math.floor(1000000*alpha2)
QuantileGaussian=SortedGaussian[IndexGauss]
#GPD 
SortedGPD=np.sort(stats.genpareto.rvs(*fit3,size=100000))
IndexGPD=math.floor(100000*alphapareto)
QuantilePareto=SortedGPD[IndexGPD]
#GEV
SortedGEV=np.sort(stats.genextreme.rvs(*fit1,size=100000))
IndexGEV=math.floor(100000*alpha)
QuantileGEV=SortedGEV[IndexGEV] 
QuantileVector=[QuantileGaussian,QuantileGEV,QuantilePareto]
print("Model Y 200 years return period  for Gaussian,GEV, and GPD are", QuantileVector)