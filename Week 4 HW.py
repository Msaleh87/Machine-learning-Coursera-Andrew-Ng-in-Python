import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

## First part of the HW ##
# make sure to comment out this input part when you uncomment the second half.
path = r'C:\Users\mahmo\OneDrive\Desktop\Data Science\ML\EX3\ex3data1'

data = loadmat(path)

# Try to understand the data, to see the dataset you are dealing with, it has X and y, already organized.
for v in data.keys():
    #print(v)

X, y = data['X'], data['y'] # X has 5000 x 400 and y is 5000 x 1
m = X.shape[0] # = y.size
X = np.hstack([np.ones((m,1)), X])
# comment up to here when you start the second half

'''
## the function to plot/show some of the data ##
def displayData(X, example_width=None, figsize=(100, 100)):
    
    m, n = X.shape # in this case m = 100 rows, and n = 400 coulmns. Each row form a picture of a number

    example_width = example_width or int(np.round(np.sqrt(n))) # That line in case the user didn't input the example width which in our case is 20
    example_height = n / example_width  # 400/20= 20. example_width and height are the dimensions of one picture which is contained in a 1 row in X

    # Compute number of items to display. In our case a grid of 10 x 10
    display_rows = int(np.floor(np.sqrt(m))) # in case square root of m is not an integer
    display_cols = int(np.ceil(m / display_rows))

    fig, ax_array = plt.subplots(display_rows, display_cols, figsize=figsize) # check out the examples at: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
    # figsize here is just for how big the frame enclosing the 10x10 items, you could delete it from the function and the command, it doesn't make much difference
    fig.subplots_adjust(wspace=0.025, hspace=0.025) # for the spacings between the images

    ax_array = [ax_array] if m == 1 else ax_array.ravel() # to make the ax_array a one dimensional vector, so that the for loop work

    for i, ax in enumerate(ax_array): # enumerate return the index number (i) and the crossponding value (ax)
        ax.imshow(X[i].reshape(example_width, example_width, order='F'), # to understand order check: https://stackoverflow.com/questions/45973722/how-does-numpy-reshape-with-order-f-work
                  cmap='Greys') # , extent=[0, 1, 0, 1]) # extent could deleted. If you don't set your axis limits, they become your extents & then don't seem to have any effect.
        ax.axis('off') # since you have this you don't need extent

# to pic a row from X and plot it, you could use: plt.imshow(X[i].reshape(20, 20, order='F'), cmap='Greys'); then: plt.show()
#rand_indices = np.random.choice(m, 100, replace=False) # Whether the sample is with or without replacement. Default is True, meaning that a value of a can be selected multiple times
sel = X[3900:4000, :]
displayData(sel)
plt.show()
'''
## Sigmoid function ##
def SG(z):
    return 1 / (1 + np.exp(-z))

## Cost Function ##
def costFunction(theta, X, y, lambda_):
    m = X.shape[0]
    #y = y.rehsape(X.shape[0],1)
    h = SG(np.dot(X,theta))
    term1 = np.dot(np.transpose((y)),np.log(h))
    term2 = np.dot(np.transpose(1-y),np.log(1-h))
    theta = theta.reshape(X.shape[1],1)
    J = (-1/m)*(term1+term2) + (0.5*lambda_/m)*sum(np.square(theta[1:,:]))
    return J

## Gradient function ##
def Gradient(theta,X,y, lambda_):
    m = X.shape[0]
    theta = theta.reshape(X.shape[1],1)
    h = SG(np.dot(X,theta))
    grad = (1/m) *np.dot(np.transpose(X),(h-y)); #alpha is not included in this line or the next
    grad[1:,:] = grad[1:,:] + (lambda_/m)* theta[1:,:]
    return grad.flatten(); # you have to use .flatten, because the op.minimize takes only 1 dimentional vector not (m,1), but (m,)! If you are using method = 'TNC' then it doesn't matter if you add .flatten or not
# you can find the source about this argumet at: https://stackoverflow.com/questions/8752169/matrices-are-not-aligned-error-python-scipy-fmin-bfgs


import scipy.optimize as op
def oneVsAll(X, y, num_labels):

    all_thetas = np.zeros([num_labels, X.shape[1]]) # 10 x 401
    for i in range(num_labels):
        print(i)
        theta = np.zeros([X.shape[1],1]) # 401 x 1 --> z = X*theta (5000 x 401) * (401 x 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = y_i.reshape(X.shape[0],1) # (5000,1)
        fmin = op.minimize(fun = costFunction,
                         x0 = theta,
                         args = (X, y_i, 0.1),
                         jac = Gradient,
                           method ='TNC')
        all_thetas[i,:] = fmin.x;
    return all_thetas

import math # for the mean function to work
# all_thetas = oneVsAll(X, y, 10) #
def predict(X, all_thetas):
    H = SG(np.dot(X,all_thetas.T)) # H is 5000 x 10, if you type H[0] you will notice that among the 1st row the first element is almost equal to 1 h(theta) = 1 meaning that this element is the closest to y = 0. H[3900], the 8th element is the closest to 1 meaning index 7 meaning most likely y = 7.
    m = X.shape[0]
    p = np.zeros((X.shape[0],1))
    for i in range(m):
        p[i] = list(H[i]).index(np.amax(H[i])) # its easier to find the index in a list, therefore, since H is 5000 x 10. The index of the max of each row represent the number in the picture.
    return p, np.mean(p == y)

'''
## The second part of the exercise ##
# make sure to comment the input from the first half, I marked it
path = r'C:\Users\mahmo\OneDrive\Desktop\Data Science\ML\EX3\ex3data1'

data = loadmat(path)

# Try to understand the data, to see the dataset you are dealing with, it has X and y, already organized.
for v in data.keys():
    print(v)

X, y = data['X'], data['y'] # X has 5000 x 400 and y is 5000 x 1
m = X.shape[0] # = y.size
X = np.hstack([np.ones((m,1)), X])
y[y == 10] = 0
# lodaing the weights #

weights = loadmat(r'C:\Users\mahmo\OneDrive\Desktop\Data Science\ML\EX3\ex3weights')
theta1 , theta2 = weights['Theta1'] , weights['Theta2']
theta2 = np.roll(theta2, 1, axis=0) # Apperantly this line is needed to put the thetas in the order that would work on python not the one from matlab. Simply this command do shift left or right, up or down based on the axis selected
# you could read more about at: https://numpy.org/doc/stable/reference/generated/numpy.roll.html

def NN(X, theta1, theta2):
    z1 = np.dot(X,theta1.T)
    A1 = SG(z1)
    A1 = np.hstack([np.ones((A1.shape[0],1)),A1])
    z2 = np.dot(A1,theta2.T)
    A2 = SG(z2)
    p = np.zeros((X.shape[0],1))
    for i in range(m):
        p[i] = list(A2[i]).index(np.amax(A2[i]))
    return p, np.mean(p == y)
'''

























    

