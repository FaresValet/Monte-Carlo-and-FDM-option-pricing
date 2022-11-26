import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import stats
import math
from datetime import datetime
from pyextremes import plot_mean_residual_life


# =============================================================================
# Importing our data
# =============================================================================
df=pd.read_excel('h:\Downloads\Données.xlsx')
data=df.loc[:,'X']
datay=df.loc[:,'Y']
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
ParetoSample=[5000*i for i in range(1,9)]
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
datapar=[x for x in data if x>=20000] #Around 2 years data-clustering
v=np.ceil(np.log2(len(data1))) + 1 #This is called Sturge's rule, this formula is used to calculate the adequate number of bins to visualize your data's distribution
v2=np.ceil(np.log2(len(data))) + 1
v3=np.ceil(np.log2(len(datapar))) + 1
y,x=np.histogram(data1,bins=int(v),density=True) #"clustering" our data for the plot
y2,x2=np.histogram(data,bins=int(v2),density=True)
y3,x3=np.histogram(datapar,bins=int(v3),density=True)
plt.hist(data1, bins=int(v), density=True)

plt.title("Histogram")
plt.show()
x = (x + np.roll(x, -1))[:-1] / 2.0
x2 = (x2 + np.roll(x2, -1))[:-1] / 2.0 
x3 = (x3 + np.roll(x3, -1))[:-1] / 2.0 #This takes the mid point of every "bin" interval as the reference x-axis point for its corresponding y probability
# =============================================================================
# Fitting our data and plotting the PDFs
# =============================================================================
fit1=stats.genextreme.fit(data1,loc=0) #The fit method finds the optimal parameters for your data fitting a chosen probability distribution
print(fit1)
fit2=stats.norm.fit(data)
fit3=stats.genpareto.fit(datapar,loc=0)
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
YaxisGaussian=np.zeros(len(Xaxis)) 
YaxisGev=np.zeros(len(Xaxis)) 
YaxisGPD=np.zeros(len(Xaxis))   
dataGaussian=stats.norm.rvs(*fit2,size=len(data)) #we generate len(data) number of values from our distributions and plot their respective probabilities
dataGPD=stats.genpareto.rvs(*fit3,size=len(data))
dataGev=stats.genextreme.rvs(*fit1,size=len(data)) 
for i in range(len(Xaxis)):
    YaxisEmpiricalDistribution[i]=empirical(data,Xaxis[i])*24
    YaxisGaussian[i]=empirical(dataGaussian,Xaxis[i])*24
    YaxisGev[i]=empirical(dataGev,Xaxis[i])*2
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
plt.plot(Xaxis,YaxisEmpiricalDistribution,c='g',label='Original Data Sample')
plt.plot(Xaxis,YaxisGPD,c='r',label='GPD Data Sample')
plt.title("Empirical distribution P(X>u)=1-P(X<u) GPD and Original Data")
plt.legend()
plt.show()
plt.plot(Xaxis,YaxisEmpiricalDistribution,c='g',label='Original Data Sample')
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
#The next part is just to avoid bias from data sample, we simulate one hundred times each data distribution and take its mean, this way we are (slightly) more accurate when it comes to validating our model
Nmc=100  
SSEVector=np.zeros((Nmc,3))
c0=0
c1=0
c2=0
for i in range(Nmc):
        dataGaussian=stats.norm.rvs(*fit2,size=len(data)) 
        dataGPD=stats.genpareto.rvs(*fit3,size=len(data))
        dataGev=stats.genextreme.rvs(*fit1,size=len(data))
        for j in range(len(Xaxis)):
            YaxisGaussian[j]=empirical(dataGaussian,Xaxis[j])*24
            YaxisGev[j]=empirical(dataGev,Xaxis[j])*2
            YaxisGPD[j]=empirical(dataGPD,Xaxis[j])
            
        SSEVector[i]=[leastsquares(YaxisGaussian),leastsquares(YaxisGPD),leastsquares(YaxisGev)]
        c0+=SSEVector[i][0]
        c1+=SSEVector[i][1]
        c2+=SSEVector[i][2]
NMCMean=[c0/Nmc,c1/Nmc,c2/Nmc] #Mean over all our simulations 

#The selected model based on this method is GPD, although this is subjective and for tail distribution GEV fits as good, we couldve chosen it as well, it is made clear though that the gaussian distribution is ruled out. In Fact I would like to add GPD and GEV are petty much equivalent for events >30000 with GEV having a slightly fatter tail, so if we want to be sure about not losing money, we would pick the GEV model for estimating return periods (because it's safer)
# =============================================================================
#  Quantiles 
# =============================================================================
#Actual Data
SortedData=np.sort(data1)
alphapareto=0.99
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
print("Model X 200 years return period  values for Gaussian,GEV, and GPD are", QuantileVector)
# =============================================================================
# X and Y Model dependency (A faire :)) 
# =============================================================================
