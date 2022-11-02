import numpy as np
import pandas as pd
from scipy.stats import norm 
import matplotlib.pyplot as plt
# =============================================================================
# Importing the cac40 data and initializing our values
# =============================================================================
df=pd.read_csv("h:\Documents\ING3Code\spx_quotedata.csv", delimiter=",")
q=0.0217
eps=0.0001
S_0=3932.69
df["S0"]=S_0
df["Marche"]=(df.iloc[:,4]+df.iloc[:,5])/2
df["Time"]=0.0
# =============================================================================
# Setting the time till maturity for every option and stocking it in df
# =============================================================================
TIndex=df.drop_duplicates(subset=['Expiration Date']).index.tolist()
TIndex.append(9677)
for i in range(1,len(TIndex)):
    for j in range(TIndex[i-1],TIndex[i]):
        df.loc[j,"Time"]=i/365
    
# =============================================================================
#    Call Function 
# =============================================================================
def Call_BS(t,S,K,T,r,sigma):
    d1=(np.log(S/K)+(r+0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
    d2=(np.log(S/K)+(r-0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))

    if t==T:
        return (np.max(S-K,0))
    else: 

        return(S*norm.cdf(d1)-K*np.exp(-r*(T-t))*norm.cdf(d2))
    

# =============================================================================
# Vega function
# =============================================================================
def Vega(t,S,K,T,r,sigma):
    d1=(np.log(S/K)+(r+0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
    return S*np.sqrt(T-t)/np.sqrt(2*np.pi)*np.exp(-d1**2/2)
# =============================================================================
# Marketprice - BS
# =============================================================================
def F(marche,t,S,K,T,r,sigma):
    return Call_BS(t, S, K, T, r, sigma)-marche
# =============================================================================
# new df columns and Newton's algorithm if no arbitrage
# =============================================================================
df.loc[:,"r"]=0.0255
df["Sigma"]=0.0
df["Strike"]=df.iloc[:,11]
t=0.0
for i in range(0,9677):
    df.loc[i,"Sigma"]=(np.sqrt(2*np.abs(np.log(df.loc[i,"S0"]/df.loc[i,"Strike"])+df.loc[i,"r"]*df.loc[i,"Time"])/df.loc[i,"Time"]))


for i in range(0,9677):
    if np.max(+df.loc[i,"S0"]-df.loc[i,"Strike"]*np.exp(-df.loc[i,"r"]*df.loc[i,"Time"]),0)<df.loc[i,"Marche"]<df.loc[i,"S0"]:
        while abs(F(df.loc[i,"Marche"],t,df.loc[i,"S0"],df.loc[i,"Strike"],df.loc[i,"Time"],df.loc[i,"r"],df.loc[i,"Sigma"]))>0.0001:    
            df.loc[i,"Sigma"]=df.loc[i,"Sigma"]-(F(df.loc[i,"Marche"],t,df.loc[i,"S0"],df.loc[i,"Strike"],df.loc[i,"Time"],df.loc[i,"r"],df.loc[i,"Sigma"]))/(Vega(t,df.loc[i,"S0"],df.loc[i,"Strike"],df.loc[i,"Time"],df.loc[i,"r"],df.loc[i,"Sigma"]))
    else:
        df.loc[i,"Sigma"]=0

# =============================================================================
# Dropping arbitrage values and plotting smiles
# =============================================================================
df = df[df.Sigma != 0]
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
ax.scatter(df.loc[:,"Time"], df.loc[:,"Strike"], df.loc[:,"Sigma"])
ax.view_init(10,50)
ax.set_ylabel('Strike')
ax.set_xlabel('Time')
ax.set_zlabel('\u03C3')
ax.set_title("Volatility Smile CAC40")
plt.show()

