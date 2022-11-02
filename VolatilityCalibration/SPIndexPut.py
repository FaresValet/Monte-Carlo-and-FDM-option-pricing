import numpy as np
import pandas as pd
from scipy.stats import norm 
import matplotlib.pyplot as plt
# =============================================================================
# Getting the sp-index put data and our initial values
# =============================================================================
df=pd.read_csv("h:\Documents\ING3Code\sp-index.txt",delimiter="\s+")
q=0.0217
eps=0.0001
S_0=1260.36
df["S0"]=np.exp(-q*df.iloc[:,0])*S_0
df["Marche"]=(df.iloc[:,4]+df.iloc[:,5])/2
# =============================================================================
# Call functon
# =============================================================================
def Call_BS(t,S,K,T,r,sigma):
    d1=(np.log(S/K)+(r+0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
    d2=(np.log(S/K)+(r-0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))

    if t==T:
        return (np.max(K-S,0))
    else: 

        return(-S*norm.cdf(-d1)+K*np.exp(-r*(T-t))*norm.cdf(-d2))
    

# =============================================================================
# Vega function
# =============================================================================
def Vega(t,S,K,T,r,sigma):
    d1=(np.log(S/K)+(r+0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
    return S*np.sqrt(T-t)/np.sqrt(2*np.pi)*np.exp(-d1**2/2)
# =============================================================================
# Market price/BS difference function for Newton's 
# =============================================================================
def F(marche,t,S,K,T,r,sigma):
    return Call_BS(t, S, K, T, r, sigma)-marche
# =============================================================================
# New columns of our dataframe to stock our data
# =============================================================================
df.loc[:,"r"]=df.loc[:,"r"]*0.01
df["Sigma"]=0.0
df["K"]=df["K"].astype(float)
t=0.0
# =============================================================================
# Initializing vol values and proceeding with Newton's algorithm if no arbitrage
# =============================================================================
for i in range(0,280):
    df.loc[i,"Sigma"]=(np.sqrt(2*np.abs(np.log(df.loc[i,"S0"]/df.loc[i,"K"])+df.loc[i,"r"]*df.loc[i,"T"])/df.loc[i,"T"]))


for i in range(0,280):
    if np.max(-df.loc[i,"S0"]+df.loc[i,"K"]*np.exp(-df.loc[i,"r"]*df.loc[i,"T"]),0)<df.loc[i,"Marche"]<df.loc[i,"K"]*np.exp(-df.loc[i,"r"]*df.loc[i,"T"]):
        while abs(F(df.loc[i,"Marche"],t,df.loc[i,"S0"],df.loc[i,"K"],df.loc[i,"T"],df.loc[i,"r"],df.loc[i,"Sigma"]))>0.0001:    
            df.loc[i,"Sigma"]=df.loc[i,"Sigma"]-(F(df.loc[i,"Marche"],t,df.loc[i,"S0"],df.loc[i,"K"],df.loc[i,"T"],df.loc[i,"r"],df.loc[i,"Sigma"]))/(Vega(t,df.loc[i,"S0"],df.loc[i,"K"],df.loc[i,"T"],df.loc[i,"r"],df.loc[i,"Sigma"]))
    else:
        df.loc[i,"Sigma"]=0
# =============================================================================
# Dropping the 0 values from our dataframe and plotting the vol smiles
# =============================================================================
df = df[df.Sigma != 0]
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
ax.scatter(df.loc[:,"T"], df.loc[:,"K"], df.loc[:,"Sigma"])
ax.view_init(10,50)
ax.set_ylabel('Strike')
ax.set_xlabel('Time')
ax.set_zlabel('\u03C3')
ax.set_title("Volatility Smile SPIndexPut")
plt.show()
plt.plot(df.loc[:57,"K"],df.loc[:57,"Sigma"])
plt.xlabel("K")
plt.ylabel("\u03C3")
plt.title("Volatility Smile for the first maturity")
plt.show()
  

