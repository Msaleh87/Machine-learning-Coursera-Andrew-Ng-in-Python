import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
#import scipy.io as scio # you could use this line but then you will have to change the data code line

path = r'add your path\ex5data1'
data = loadmat(path)
#data = scio.loadmat(path) # in case you use the commented import command.

#print(type(data)) # to know what type of data you are dealing with

'''
# to know what's inside this dictionary
for keys in data.keys():
    print(keys)
    print(len(data[keys])) #check out how to know about dictionary size
'''
# assign the data to arrays in numpy
X , y , Xtest, ytest, Xval, yval = data['X'], data['y'], data['Xtest'], data['ytest'], data['Xval'], data ['yval']

'''
# visualize the data
plt.plot(X,y,'+')
plt.show()
'''
XX = np.copy(X)
XXval = np.copy(Xval)
#theta = np.ones((2,1))
X = np.hstack([np.ones((X.shape[0],1)),X])
Xval = np.hstack([np.ones((Xval.shape[0],1)),Xval])

def CostFunc(X,y, theta, L):
    m = X.shape[0]
    theta = theta.reshape(X.shape[1],1);
    y = y.reshape((m,1));
    h = np.dot(X,theta) # (12,2) x (2,1)
    J = (sum((h - y)**2))/(2*m) + (0.5*L/m)*sum(theta[1:,:]**2)
    grad = np.dot(X.T,(h-y))*(1/m) # (2,12) x (12,1) = (2,1)
    #print(grad)
    grad[1:,:] = grad[1:,:] + (L/m)*theta[1:,:]
    return J, grad.flatten()

import scipy.optimize as op
from scipy.optimize import fmin_cg


 # Creats a function that takes an input theta and return CostFunc(X,y, theta)

'''
def E(theta):
    return CostFunc(X,y, theta, 0)



#E = lambda theta: CostFunc(X,y, theta, 0)
results = op.minimize(E,theta, method = 'TNC', jac = True, options={'maxiter':100, "disp":True})

# print(op.minimize(E,theta, method = 'TNC', jac = True, options={'maxiter':100, "disp":True}))
# jac : bool or callable, optional Jacobian (gradient) of objective function. Only for CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg.
# If jac is a Boolean and is True, fun is assumed to return the gradient along with the objective function. If False, the gradient will be estimated numerically.

print(results.x)
oth = results.x


plt.plot(X[:,1],np.dot(X,oth))
plt.plot(X[:,1],y,'o')
plt.show()
'''



def LerningCurves(X,y,Xval,yval,L):
    m = X.shape[0]
    J_train = np.zeros((m,1))
    J_val = np.zeros((m,1))
    theta = np.zeros((X.shape[1],1))
    #J_m = np.zeros((m,1))
    for i in range(1,m+1): # we started from i = 1 so that the first training set has two samples, it doesn't make sense to start from zero and have 1 sample
        E = lambda theta: CostFunc(X[:i,:],y[:i,:], theta, L)[0]  # [0] if you are using fmin_cg
        results = op.fmin_cg(E,theta, maxiter = 55)
        #results = op.minimize(E,theta, method = 'TNC', jac = True, options={'maxiter':200,'disp':True}) # if you remove method you get something similar to the pdf
        #print(results)
        J_train[i-1] = CostFunc(X[:i,:],y[:i,:], results, 0)[0]
        J_val[i-1] = CostFunc(Xval,yval, results, 0)[0]   # remember to add a column of ones to Xval as well, otherwise you will get an error during computing the error
        #J_m[i-1] = i # just a counter
    J_m = np.arange(1, X.shape[0]+1)
    #J_mval = np.arange(1, Xval.shape[0]+1)
    return J_train, J_val, J_m #, J_mval


'''
theta = np.ones((X.shape[1],1))
J_T, J_V, J_m = LerningCurves(X,y,Xval,yval, 0)
plt.plot(J_m,J_T)
plt.plot(J_m,J_V)
plt.xlabel('Number of training samples')
plt.ylabel('Error')
plt.yticks(np.arange(0, 200, 20))
plt.grid()
plt.show()
'''

# everything up to here seems to be working perfectly // Check for an online code and see if any would give similar results to the pdf, if your doubt about the nuemerical method is true then non online code shall give same answer to the pdf unless they are using specific method
#x = (x-np.mean(x))/np.std(x)


def polyFeatures(X, p):
    m = X.shape[0]
    X_poly = np.zeros((m,p))
    for i in range(0,p):
        X_poly[:,i]= X[:,0]**(i+1)
    return X_poly
def featureNormalize(X):
    mu = np.mean(X, axis=0)
    X_norm = X - mu

    sigma = np.std(X_norm, axis=0, ddof=1)
    X_norm /= sigma
    return X_norm, mu, sigma

p = 8
X_poly = polyFeatures(XX, p)
X_poly, mu, sigma = featureNormalize(X_poly)
X_poly = np.concatenate([np.ones((X_poly.shape[0], 1)), X_poly], axis=1)
# mapping the single feature to the power of eight #



X_8 = polyFeatures(XX, p)
X_8, mu, sigma = featureNormalize(X_8)
X_8 = np.concatenate([np.ones((X_8.shape[0], 1)), X_8], axis=1)
Xval_8 = polyFeatures(XXval, p)
Xval_8 -= mu
Xval_8 /= sigma
Xval_8 = np.concatenate([np.ones((yval.size, 1)), Xval_8], axis=1)


### Up to here all is reviewed ## Also make the code work with 1 X not XX and copy, just in case to avoid any unexpected things
theta = np.zeros((X_8.shape[1],1)) # intializtion of thetas
'''
J_T, J_V, J_m = LerningCurves(X_8,y,Xval_8,yval, 100)

plt.plot(J_m,J_T)
plt.plot(J_m,J_V)
plt.xlabel('Number of training samples')
plt.ylabel('Error')
plt.yticks(np.arange(0, 100, 10))
plt.grid()
plt.show()
'''



'''
# get the optimal thetas for X power 8 #
K = lambda theta: CostFunc(X_8,y, theta, 100)[0]
results = op.fmin_cg(K,theta, maxiter = 55)
#results = op.minimize(K,theta, method = 'TNC',jac = True ,options={'maxiter':200}) ## add the method and number of iteration back
Opt_theta = results


min_x = np.min(X[:,1])
max_x = np.max(X[:,1])
xx_8 = np.arange(min_x - 15, max_x + 25, 0.05).reshape(-1,1)
#xx_8 = np.hstack([np.ones((xx_8.shape[0],1)),xx_8])
x_8 = polyFeatures(xx_8, p)
x_8 -= mu
x_8 /= sigma
x_8 = np.concatenate([np.ones((x_8.shape[0], 1)), x_8], axis=1)


plt.plot(xx_8[:,0],np.dot(x_8,Opt_theta), '--')
plt.plot(X[:,1],y,'x')
#plt.yticks(np.arange(0, 80, 20))
plt.grid()
plt.show()
'''

def LvsError(X,y,Xval,yval):
    lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    J_T = np.zeros(len(lambda_vec ))
    J_V = np.zeros(len(lambda_vec))
    theta = np.zeros((X.shape[1],1))
    for i in range(len(lambda_vec)):
        E = lambda theta: CostFunc(X,y, theta, lambda_vec[i])[0]  # [0] if you are using fmin_cg
        results = op.fmin_cg(E,theta, maxiter = 55)
        #results = op.minimize(E,theta, method = 'TNC', jac = True, options={'maxiter':200,'disp':True}) # if you remove method you get something similar to the pdf
        #print(results)
        J_T[i] = CostFunc(X,y, results, 0)[0]
        J_V[i] = CostFunc(Xval,yval, results, 0)[0]
    return J_T, J_V, lambda_vec
'''
J_T, J_V, lambda_vec = LvsError(X_8,y,Xval_8,yval)

plt.plot(lambda_vec,J_T)
plt.plot(lambda_vec,J_V)
plt.xlabel('LAMBDA')
plt.ylabel('Error')
plt.yticks(np.arange(0, 20, 2))
plt.grid()
plt.show()
'''

'''
XT_8 = polyFeatures(Xtest, p)
XT_8 -= mu
XT_8 /= sigma
XT_8 = np.concatenate([np.ones((ytest.size, 1)), XT_8], axis=1)

theta = np.zeros((X_8.shape[1],1))
E = lambda theta: CostFunc(X_8,y, theta, 3)[0]  # you train for lambda = 3
OP_3 = op.fmin_cg(E,theta, maxiter = 55)
J_Test = CostFunc(XT_8,ytest, OP_3, 0)[0] # when you compute the error, you do it with lambda. lambda is only used during the training to adjust thetas to reduce over fitting
'''

import random
def LerningCurves_50(X,y,Xval,yval,L):
    m = X.shape[0]
    J_train = np.zeros((m,1))
    J_val = np.zeros((m,1))
    
    #J_m = np.zeros((m,1))
    for j in range(2,m+1): # we started from i = 1 so that the first training set has two samples, it doesn't make sense to start from zero and have 1 sample
        print(j)
        for i in range(0,50):
            #print(i)
            R1 = random.sample(range(m),j)
            R2 = random.sample(range(Xval.shape[0]),j)
            JT = 0
            JV = 0
            theta = np.zeros((X.shape[1],1))
            E = lambda theta: CostFunc(X[R1,:],y[R1,:], theta, L)[0];  # [0] if you are using fmin_cg
            results = op.fmin_cg(E,theta, maxiter = 100, disp = False);
            #results = op.minimize(E,theta, method = 'TNC', jac = True, options={'maxiter':200})
            JT = JT + CostFunc(X[R1,:],y[R1,:], results, 0)[0]
            JV = JV + CostFunc(Xval,yval, results, 0)[0]
        #print('ooooooooooooooooooooooooooooo')
        #print(JT)
        J_train[j-1] = JT
        J_val[j-1] = JV   # remember to add a column of ones to Xval as well, otherwise you will get an error during computing the error
        #J_m[i-1] = i # just a counter
    J_m = np.arange(1, X.shape[0]+1)
    #J_mval = np.arange(1, Xval.shape[0]+1)
    return J_train, J_val, J_m #, J_mval

J_TT, J_VV, J_m = LerningCurves_50(X_8,y,Xval_8,yval, 0.01)

plt.plot(J_m,J_TT[1:,:])
plt.plot(J_m,J_VV[1:,:])
plt.xlabel('Number of training samples')
plt.ylabel('Error')
#plt.yticks(np.arange(0, 100, 10))
plt.grid()
plt.show()
