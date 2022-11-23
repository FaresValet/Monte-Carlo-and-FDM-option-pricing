import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import stats
# =============================================================================
# Reading the file and choosing a set number of bins 
# =============================================================================
df=pd.read_excel('h:\Downloads\Données.xlsx')
data=df.loc[:,'Y']
data = df[["date", "Y"]]
data = data.set_index("date")
data1=data.resample("Y").max()
datapar=[x for x in df.loc[:,'Y']  if x>=3000]
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
fit3=stats.genpareto.fit(datapar,loc=5)
fit4=stats.weibull_min.fit(data,loc=0) 
fit5=stats.gumbel_r.fit(data1,loc=0)
pdf=stats.genextreme.pdf(x2,*fit1)
pdf2=stats.norm.pdf(x,*fit2)
pdf3=stats.genpareto.pdf(x3,*fit3)
pdf4=stats.weibull_min.pdf(x,*fit4)
pdf5=stats.gumbel_r.pdf(x2,*fit5)
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
plt.plot(x,pdf4,c='y',label='Weibull')
plt.plot(x,y,c='g',label='Real Data ')
plt.legend()
plt.title(" Data and Weibull probability densities")
plt.show()
plt.plot(x2,pdf5,c='b',label='Gumbel Right Law')
plt.plot(x2,y2,c='g',label='Real Data ')
plt.legend()
plt.title(" Data and Gumbel Right Skew densities")
plt.show()
# =============================================================================
# Finding the best fit using least-squares
# =============================================================================
Test=[0]*4
Distributions=[pdf,pdf2,pdf3,pdf4] 
for i in range(4): #We use least squares to find which distribution from GEV/GPD/Gaussian fits our data the "best"
 Test[i]= np.sum((y - Distributions[i])**2)
 

