import numpy as np
import pandas as pd
from scipy.stats import norm 
import matplotlib.pyplot as plt
# =============================================================================
# Data and parameters
# =============================================================================
df=pd.read_excel("h:\Documents\ING3Code\GoogleOrig.xlsx")
q=0.0217
eps=0.0001
S_0=591.66
df["S0"]=S_0
df["Marche"]=df.iloc[:,2]
# =============================================================================
# Call Function
# =============================================================================
def Call_BS(t,S,K,T,r,sigma):
    d1=(np.log(S/K)+(r+0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
    d2=(np.log(S/K)+(r-0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))

    if t==T:
        return (np.max(S-K,0))
    else: 

        return(S*norm.cdf(d1)-K*np.exp(-r*(T-t))*norm.cdf(d2))
    

# =============================================================================
# Vega Function
# =============================================================================
def Vega(t,S,K,T,r,sigma):
    d1=(np.log(S/K)+(r+0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
    return S*np.sqrt(T-t)/np.sqrt(2*np.pi)*np.exp(-d1**2/2)
# =============================================================================
# MarketPrice - BS
# =============================================================================
def F(marche,t,S,K,T,r,sigma):
    return Call_BS(t, S, K, T, r, sigma)-marche

df.loc[:,"r"]=0.06
df["Sigma"]=0.0
df["Strike"]=df["Strike"].astype(float)
t=0.0
# =============================================================================
# Newton's algorithm if no arbitrage
# =============================================================================
for i in range(0,209):
    df.loc[i,"Sigma"]=(np.sqrt(2*np.abs(np.log(df.loc[i,"S0"]/df.loc[i,"Strike"])+df.loc[i,"r"]*df.loc[i,"Time"])/df.loc[i,"Time"]))


for i in range(0,209):
    if np.max(+df.loc[i,"S0"]-df.loc[i,"Strike"]*np.exp(-df.loc[i,"r"]*df.loc[i,"Time"]),0)<df.loc[i,"Marche"]<df.loc[i,"S0"]:
        while abs(F(df.loc[i,"Marche"],t,df.loc[i,"S0"],df.loc[i,"Strike"],df.loc[i,"Time"],df.loc[i,"r"],df.loc[i,"Sigma"]))>0.0001:    
            df.loc[i,"Sigma"]=df.loc[i,"Sigma"]-(F(df.loc[i,"Marche"],t,df.loc[i,"S0"],df.loc[i,"Strike"],df.loc[i,"Time"],df.loc[i,"r"],df.loc[i,"Sigma"]))/(Vega(t,df.loc[i,"S0"],df.loc[i,"Strike"],df.loc[i,"Time"],df.loc[i,"r"],df.loc[i,"Sigma"]))
    else:
        df.loc[i,"Sigma"]=0


# =============================================================================
# Dropping arbitrage values and plotting smiles for Google index data
# =============================================================================
df = df[df.Sigma != 0]
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
ax.scatter(df.loc[:,"Time"], df.loc[:,"Strike"], df.loc[:,"Sigma"])
ax.view_init(10,50)
ax.set_ylabel('Strike')
ax.set_xlabel('Time')
ax.set_zlabel('\u03C3')
ax.set_title("Volatility Smile Google")
plt.show()
