import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# =============================================================================
# Basic linear/polynomial regression template with gradient descent
# =============================================================================

x,y=make_regression(n_samples=100,n_features=1,noise=10)
y=y.reshape(y.shape[0],1)
y=y**3
X=np.hstack((x**3,x**2,x,np.ones((x.shape))))
theta=np.random.randn(4,1)

def function(X,theta):
    return X.dot(theta)
def CostFunction(X,y,theta):
    m=len(y)
    return 1/(2*m)*np.sum((function(X,theta)-y)**2)
def gradient(X,y,theta):
    m=len(y)
    return 1/m * X.T.dot(function(X,theta)-y)
def gradientdescent(X,y,theta,alpha,itera):
    cost=np.zeros(itera)
    for i in range(itera):
        theta=theta-alpha*gradient(X,y,theta)
        cost[i]=CostFunction(X,y,theta)
    
    return theta, cost

pred,costi=gradientdescent(X,y,theta,0.01,500)

plt.scatter(x,y)
plt.scatter(x,function(X,pred),c='r')
plt.xlabel('Input Data')
plt.ylabel('Output Data')
plt.title('Exact data in blue and model prediction in red')
plt.show()
"""fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(x[:,0],x[:,1],y)
ax.scatter(x[:,0],x[:,1],function(X,pred))"""