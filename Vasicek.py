
# =============================================================================
# We'll be simulating interest rate paths using the Vasicek model
# Interest rates are simulated using an Ornstein-Uhlenbeck Process
# Contrarely to stock prices Interest rates do not rise forever but tend to "return" to some value after a while otherwise it may stunt the economy (if they kept rising)
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
def vasicek(r,K,theta,sigma,T,N,seed=0):
    np.random.seed(seed)  #Generating the same random numbers
    dt=T/N
    Interest_rates=[r]
    for i in range(N):
        dr=K*(theta-Interest_rates[-1])*dt+sigma*np.random.normal()
        Interest_rates.append(Interest_rates[-1]+dr)
    return Interest_rates

x= vasicek(0.01875, 0.20, 0.01, 0.012, 20, 200)
x=np.array(x)
plt.plot(x)
plt.title('Interest rate path Vasicek model')