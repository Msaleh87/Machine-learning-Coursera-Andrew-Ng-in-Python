import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.optimize as op

# an r was added to the path so that it wuld be read raw, otherwise '\' will be considered a new line and you will get an error
# OR: you replace every '\' with '\\' manually
# change the path to the path where the HW data is on your PC
path = r'C:\Users\OneDrive\Desktop\Data Science\ML\EX2\ex2data2.txt'
data = np.loadtxt(path,delimiter=',')


NF = 2  # number of features, enter it manually.
m = data.shape[0] # number of examples you have.
x = data[:, :NF] # picking up the features and put them in X.
y = data[:, NF].reshape(m,1)
# picking up the output and put it in y.
# without reshaping y, the output matrix/vector will be (m,),  which will give you error dimension through out the code.

'''
## the code for plotting ##
##-----------------------##
pos= np.where(y==1)[0] # it returns the element order in that vector when y = 1
neg = np.where(y==0)[0] # t returns the element order in that vector when y = 0
# now, we want to find the cordinates of x1 and x2 that crossponds to y = 1
# since x1 = data[:,0] and x2 = data[:,1] and we have pos, then data[pos, 0] returns the values of x1 when y =1 and data[pos, 1] returns the values of x2 when y = 1.
plt.plot(data[pos, 0], data[pos, 1], 'b*')
# The cordinates of x1 and x2 that crossponds to y = 0
plt.plot(data[neg, 0], data[neg, 1], 'ro')
plt.show()
'''

## mapping feature code##
# polynomial from the 6th degree, the code below is suppose to do the following:
# at i = 1, j = 0,1 ---> x1,x2
# at i = 2, j = 0,1,2 ---> x1^2, x1x2, x2^2
# at i = 3, j = 0,1,2,3 ---> x1^3, x1^2*x2, x1*x2^2, x2^3 and so untill
# at i = 6, j = 0,1,2,3,4,5,6 ---> x1^6, x1^5*x2, x1^4*x2^2, x1^3*x2^3, x1^2*x2^4, x1*x2^5, x2^6
# this shall add up to 27 terms and then you add the ones colomn and you get 28 terms/features


degree = 6 # the degree of the polynomial of your choice
X = np.ones((m,1))

for i in range(1,degree+1):
    # print('loop i = ' + str(i)) # in case you want to see the mapping as it happens
    for j in range(i+1):
        #print('j = ' + str(j))
        #print('X1^' + str (i-j) + ' *X2^' + str(j)) # to see the terms of the polynomial I wrote up there
        term1 = np.power(x[:,0],(i-j))
        term2 = np.power(x[:,1],j)
        K = np.multiply(term1,term2).reshape(m,1) # Again the reshape to change from (m,) to (m,1). Otherwise you will have an error when it comes to np.hstack cuz of dimensions
        # K = x[:,0]**(i-j)*x[:,1]**j # this could be used as a fast alternative
        # K = K.reshape(m,1)
        X = np.hstack([X , K])

theta=np.ones((X.shape[1],1)) # choose the intial values of your thetas. The number of features = X.shape[1] = n after the feature mapping which is 28

## Sigmoid function ##
def SG(z):
    return 1 / (1 + np.exp(-z))

## Cost Function ##
def costFunction(theta, X, y, lambda_):
    #print('Theta shape = ' + str(theta.shape)) # You will find that the optimization function passes theta with a shape of (n,), thats why we have to reshape it
    h = SG(np.dot(X,theta))
    term1 = np.dot(np.transpose((y)),np.log(h))
    term2 = np.dot(np.transpose(1-y),np.log(1-h))
    theta = theta.reshape(X.shape[1],1)
    J = (-1/m)*(term1+term2) + (0.5*lambda_/m)*sum(np.square(theta[1:,:]))
    # notice in the baised term of J that we count thetas starting from j=1 not 0 because we are requalizing the features and theta0 has no feature
    '''
    # Akternative to calculate J using sum instead of the vectorized version up there
    k = 1-y
    l = 1-h
    J = (-1/m)*sum(np.multiply(y,np.log(h))+np.multiply(k,np.log(l))) + (0.5*lambda_/m)*sum(np.square(theta[1:,:]))
    '''
    return J

## Gradient function ##
def Gradient(theta,X,y, lambda_):
    theta = theta.reshape(X.shape[1],1)
    h = SG(np.dot(X,theta))
    '''
    # in case you want to see how many times the optimization function calls this funtion you built. Just a simple counter. i was made global so that it doesn't get erased after the function is done
    global i
    i = i + 1
    print(i)
    '''
    grad = (1/m) *np.dot(np.transpose(X),(h-y));
    # Alternative
    #grad = ((x.T).dot(h-y))/m;
    grad[1:,:] = grad[1:,:] + (lambda_/m)* theta[1:,:] # add the bias term to all the elements except for theta0
    return grad;

## Optimization function from Scipy ##
# this function calls/uses all the funtions you built up there. args = ((X, y, 1)), where lambda = 1, you could change that if you want.
Result = op.minimize(fun = costFunction, 
                                 x0 = theta, # uses theta as an intial value, we sat it to ones up there, you could change it to zeros if you want
                                 args = ((X, y, 1)), # Mapped features, output, and lambda
                                 method = 'TNC', # the method used 
                                 jac = Gradient, # it call the function gradient you built
                                 options = {'maxiter':100}); # you could change the number of iteration here
oth = Result.x; # returns the optimal theta that will reduce the cost function J
cost = Result.fun; # returns the cost at the optimal thetas

def predict(X,theta):
    h = SG(np.dot(X,theta)); # Calculate the Sigmoid h = g(z)
    p=[0]; # a dummy array so that the vstack command works, you could delete later like in the comment below or just use p[1:,:], which excludes the dummy element
    for i in h:
        if i >= 0.5: # if h >= 0.5 --> P(y=1|x;theta) = 1
            p = np.vstack([p,1])
        else:       # # if h < 0.5 --> P(y=0|x;theta) = 0
            p = np.vstack([p,0])
    '''
    #
    p = np.delete(p,0)
    p = p.reshape(X.shape[0],1)
    return p.shape, np.mean(p == y)*100
    '''
    return p[1:,:], p[1:,:].shape, np.mean(p[1:,:] == y)*100 # Compare your optimzation to the actual output of the data set y

# print the prediction function using the mapped features from the original features and the optimal calculated theta from the optimization function
# Note, if you use a different set of data X, np.mean(p[1:,:] == y)*100 will mean nothing in that case. This was specifically done to compute the accuracy of your optimization
# The prediction in case of new set of data X would be p[1:,:] only, you would need to comment the rest

print(predict(X,oth))

## Mapping feature function ##
# similar to the one above, but this was made as a function to be used in the plotting
def MapFeatures(x1,x2):
    m = x1.size
    L = np.ones((m,1))
    for i in range(1,degree+1):
        #print('loop i = ' + str(i))
        for j in range(i+1):
            #print('X1^' + str (i-j) + ' *X2^' + str(j))
            term1 = x1**(i-j)
            term2 = x2**j
            K = np.multiply(term1,term2).reshape(m,1)
            #print(K)
            L = np.hstack([L , K])
    return L


## Plotting the boundary decision ##

# bulding a grid/matrix of 50 by 50 elements, not sure why the limits were choosen -1 and 1.5
u = np.linspace(-1,1.5,50)
v = np.linspace(-1,1.5,50)
z = np.zeros((len(u),len(v)))
for i in range(len(u)):
    for j in range(len(v)):
        z[i,j]= np.dot(MapFeatures(u[i], v[j]),oth) # now take every element in the matrix and compute Theta transpose * X, which is z
z = z.T 
plt.contour(u, v, z)

pos= np.where(y==1)[0]
neg = np.where(y==0)[0]
plt.plot(data[pos, 0], data[pos, 1], 'b*')
plt.plot(data[neg, 0], data[neg, 1], 'ro')
plt.show()

# if you want to see different decision boundary, change lambda in the optimization function
        

