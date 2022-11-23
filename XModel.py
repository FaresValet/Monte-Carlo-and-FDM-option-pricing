import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import stats
import math
from datetime import datetime
# =============================================================================
# Reading the file and choosing a set number of bins 
# =============================================================================
df=pd.read_excel('h:\Downloads\Données.xlsx')
data=df.loc[:,'X']
datay=df.loc[:,'Y']
data1 = df[["date", "X"]]
data1 = data1.set_index("date")
data1=data1.resample("Y").max()
datapar=[x for x in data if x>=20000]
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
fit1=stats.genextreme.fit(data1,loc=0,scale=1) #The fit method finds the optimal parameters for your data fitting a chosen probability distribution
fit2=stats.norm.fit(data)
fit3=stats.genpareto.fit(datapar,loc=0)
fit4=stats.weibull_min.fit(data1,loc=0) 
fit5=stats.exponweib.fit(data1,loc=0)
fit6=stats.gumbel_r.fit(data1,loc=0)
fit7=stats.gumbel_l.fit(data1,loc=0)
pdf=stats.genextreme.pdf(x,*fit1)
pdf2=stats.norm.pdf(x2,*fit2)
pdf3=stats.genpareto.pdf(x3,*fit3)
pdf4=stats.weibull_min.pdf(x,*fit4)
pdf5=stats.exponweib.pdf(x,*fit5)
pdf6=stats.gumbel_r.pdf(x,*fit6)
pdf7=stats.gumbel_l.pdf(x,*fit7)
plt.plot(x2,pdf2,c='r',label='Gaussian Law')
plt.plot(x2,y2,c='g',label='Real Data ')
plt.xlim(20000,80000)
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
plt.plot(x,pdf4,c='b',label='Weibull Law')
plt.plot(x,y,c='g',label='Real Data ')
plt.legend()
plt.title(" Data and Weibull densities")
plt.show()
plt.plot(x,pdf5,c='b',label='Expon weibull Law')
plt.plot(x,y,c='g',label='Real Data ')
plt.legend()
plt.title(" Data and Expon Weibull densities")
plt.show()
plt.plot(x,pdf6,c='b',label='Gumbel Right Law')
plt.plot(x,y,c='g',label='Real Data ')
plt.legend()
plt.title(" Data and Gumbel Right Skew densities")
plt.show()
plt.plot(x,pdf7,c='b',label='Gumbel left Law')
plt.plot(x,y,c='g',label='Real Data ')
plt.legend()
plt.title(" Data and Gumbel Left sweked densities")
plt.show()
# =============================================================================
# Finding the best fit using least-squares
# =============================================================================

# =============================================================================
#  Quantiles 
# =============================================================================
#Actual Data
SortedData=np.sort(data1)
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
IndexGPD=math.floor(100000*alpha)
QuantilePareto=SortedGPD[IndexGPD]
#GEV
SortedGEV=np.sort(stats.genextreme.rvs(*fit1,size=100000))
IndexGEV=math.floor(100000*alpha)
QuantileGEV=SortedGEV[IndexGEV]
#Weibull
SortedWeibull=np.sort(stats.weibull_min.rvs(*fit4,size=100000))
IndexWeibull=math.floor(100000*alpha)
QuantileWeibull=SortedWeibull[IndexWeibull]
QuantileVector=[QuantileData,QuantileGaussian,QuantilePareto,QuantileGEV,QuantileWeibull] 
#ExponentialWeibull
SortedExpoWeibull=np.sort(stats.exponweib.rvs(*fit5,size=100000))
IndexExpoWeibull=math.floor(100000*alpha)
QuantileExpoWeibull=SortedExpoWeibull[IndexExpoWeibull]
QuantileVector=[QuantileData,QuantileGaussian,QuantilePareto,QuantileGEV,QuantileWeibull,QuantileExpoWeibull] 
# =============================================================================
# Dépendance Des deux modèles X et Y
# =============================================================================
plt.scatter(data,datay)
plt.show()