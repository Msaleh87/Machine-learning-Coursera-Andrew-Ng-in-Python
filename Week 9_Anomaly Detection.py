import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

data = loadmat(r'PUT YOUR PATH\ex8data1')

for keys in data.keys():
    print(keys)
X , Xval, yval = data['X'], data['Xval'], data['yval']

'''
plt.plot(X[:,0],X[:,1], 'bx')
plt.show()
'''
## 1.2 Estimating parameters for a Gaussian ##

def estimateGaussian(X):
    mu = np.mean(X,axis = 0)
    sigma = np.std(X,axis = 0, ddof =0)**2
    return mu, sigma

import numpy.linalg

def multivariateGaussian(X, mu, Sigma):
    n = mu.size
    if Sigma.ndim == 1:
        Sigma = np.diag(Sigma)
    X = X - mu
    
    C = (1/((2*np.pi)**(n/2) * np.linalg.det(Sigma)**(0.5))) # the constant before the exponential
    P = C*np.exp(-0.5*np.sum(np.dot(X,np.linalg.pinv(Sigma))*X, axis=1)) # This is the vectorized version of the multivariat Gaussian formula in the lecture of a shape (m,)
    # If you take one example of the m set example then inside the exponential you would have, for n = 2 features, [x1-u1 x2-u2]* [[1/sigma1^2 0],[0 1/sigma2^2]] .* [x1-u1 x2-u2] = [1/sigma1^2*(x1-u1)^2 1/sigma2^2*(x2-u2)^2]
    # this is the expected results according to the lecture (page 4) at: https://cs229.stanford.edu/section/gaussians.pdf
    return P.reshape(X.shape[0],1)


def visualizeFit(X, mu, sigma):
    X1, X2 = np.meshgrid(np.arange(0, 35.5, 0.5), np.arange(0, 35.5, 0.5)) # np.arange(0, 35.5, 0.5) is values from 0 to 35 with an increment of 0.5, it has a shape of 71
    # The purpose of meshgrid is to create a rectangular grid out of an array of x values and an array of y values. In our example, its creating a grid of 71x71 elements
    # The reason we choose 0 and 35.5 to be our range is because min(X[:,0]) and max(X[:,0]) are abot 4 and 24. Similarly with X[:,1]
    # If you type X1[0] or X1[1] .. X[70] you would get [0 0.5 ... 35], X2[0] = all zeros, X2[1]= all 0.5, X2[2] = all 1. So that you would be able to have all combinations
    # If you feel you want to explore the meshgrid more, check out this: https://stackoverflow.com/questions/36013063/what-is-the-purpose-of-meshgrid-in-python-numpy
    Z = multivariateGaussian(np.stack([X1.ravel(), X2.ravel()], axis=1), mu, sigma) # now we want to calculate the probability of each combination of X1 and X2
    # We will stack X1.ravel() and X2.ravel()] in two coulmns (axis=1).
    # In this case we are passing to the multivariateGaussian func an X of shape (71x71=5041,2) that has all the combinations of the grid created earlier
    # We are using the same mu and Sigma we acquired before because we want the graph to be shaped around these values
    Z = Z.reshape(X1.shape) # now we want the values/probiblity of each combination to have the same shape of the grid 71x71

    plt.plot(X[:, 0], X[:, 1], 'bx', mec='b', mew=2, ms=8) # plot the orignal values of X data

    if np.all(abs(Z) != np.inf):
        plt.contour(X1, X2, Z, levels=10**(np.arange(-20., 1, 3)), zorder = 100) # Levels represent the number of countors' circles and how big they are.
        # Over here np.arange(-20., 1, 3) gives 7 levels 10^[-20., -17., -14., -11.,  -8.,  -5.,  -2.]
        # zorder, if it's negative, it means that the countor circles will be drawn underneath the scattered points, if its positive it will be drawn above, try 100 and -100 and see the difference
    plt.show()
    
mu, sigma = estimateGaussian(X)
visualizeFit(X, mu, sigma)

## 1.3 Selecting the threshold ##

pval = multivariateGaussian(Xval, mu, sigma) # make sure that pval and yval have the exact same shape, otherwise you will get weird results

def selectThreshold(pval,yval):
    Fscore = []
    eps = []
    for epsilon in np.linspace(1.01*min(pval),max(pval),1000):
        #print(epsilon)
        Fp = np.sum((pval < epsilon)& (yval == 0)) #classify a sample as anomoly while it's not
        # pval < epsilon (prediction is anomoly) , yval == 0 is (actual value is not an anamoly)
        Tp = np.sum((pval < epsilon)& (yval == 1)) #classify a sample as anomoly while it is
        # yval == 1 (actual value is an anomoly)
        Fn = np.sum((pval > epsilon)& (yval == 1)) #classify a sample as not anomoly while it is
        # pval > epsilon (predicts to be not an anomoly)
        prec = Tp/(Tp+Fp) 
        rec = Tp/(Tp+Fn)
        F1 = (2*prec*rec)/(prec+rec)   
        Fscore.append(F1) # stores F1 scores in Fscore
        eps.append(epsilon) # and store the respective epsilon to the above F1 score
    F = max(Fscore) # Find the max/ best F1 score 
    E = eps[Fscore.index(max(Fscore))] # Fscore.index(max(Fscore)) gives the index order of the "max" F1 score in Fscore then eps[this index] gives the crossponding/respective epsilon to the max F1 score
    return E, F 

print('For data set 1, Epsilon and F1 score are: ')
print(selectThreshold(pval,yval))

  
## 1.4 High dimensional dataset ##

data2 = loadmat(r'PUT YOUR PATH\ex8data2')

X , Xval, yval = data2['X'], data2['Xval'], data2['yval']
mu, sigma = estimateGaussian(X)
pval = multivariateGaussian(Xval, mu, sigma)
print('For data set 2, Epsilon and F1 score are: ')
print(selectThreshold(pval,yval))


