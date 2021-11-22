import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

## loading the data ##

path = r'C:\Users\mahmo\OneDrive\Desktop\Data Science\ML\EX4\ex4data1'
data = loadmat(path)

for v in data.keys():
    print(v)

X, y = data['X'], data['y']
m = X.shape[0]
X = np.hstack([np.ones((m,1)),X])
y[y==10] = 0

## loading the weights ##

weights = loadmat(r'C:\Users\mahmo\OneDrive\Desktop\Data Science\ML\EX4\ex4weights')
Theta1, Theta2 = weights['Theta1'], weights['Theta2']
Theta2 = np.roll(Theta2, 1, axis=0)
nn_params = np.concatenate([Theta1.ravel(),Theta2.ravel()])

def SG(z):
    return 1 / (1 + np.exp(-z))

KK = 0
def NN(X, y, nn_params):   # , num_labels, hidden_LSize,
    m = X.shape[0]
    ## getting the theates from nn_params
    Theta1 = nn_params[0:(25*401)].reshape(25,401) # (25,401)
    Theta2 = nn_params[(25*401):((25*401)+(10*26))].reshape(10,26) # (10,26))
    ## start of the feedforward
    z2 = np.dot(X,Theta1.T)
    A2 = SG(z2)
    A_2=np.copy(A2)
    A2 = np.hstack([np.ones((A2.shape[0],1)),A2]) #(5000,26)
    z3 = np.dot(A2,Theta2.T)
    A3 = SG(z3) #(5000,10)

    Y = np.zeros((m,10))                # (5000,10)
    J = 0  # cost function counter
    ## Intializing the values to start the Bp
    dz3 = np.zeros((5000,10)) # A3[i](10,) - Y[i](10,)  (10,)
    dz2 = np.zeros((25,)) # Theta2.T(26,10) * dz3(10,)[1:](25,) .* SGGrad(z2)(25,) = (25,)
    delta2 = np.zeros((10,26)) # dz3(10,1)*A2[i].T(1,26) = (10,26)
    delta1 = np.zeros((25,401)) # dz2(25,1)*A1[i].T(1,401) = (25,401)
    
    for i in range(m):
        Y[i][y[i]] = 1
        for k in range(10):
            J = J + Y[i][k]*np.log(A3[i][k]) + (1-Y[i][k])*np.log(1-A3[i][k])
        ## up to here the feedforward is done and the J value is the same as the one in the PDF
    dz3 = A3 - Y #(5000,10)
    GZ2_prime = A_2*(1-A_2) #(5000,25)
    dz2 = np.multiply(np.dot(dz3,Theta2[:,1:]),GZ2_prime) # (5000,25)

    delta2 = delta2 + np.dot(dz3.T,A2) # (5000,10).T , (5000,26)
    delta1 = delta1 + np.dot(dz2.T,X)  # (5000,25).T , (5000,401) 
    lamda = 1
    delta1[1:,:] = delta1[1:,:] + (lamda * Theta1[1:,:])
    delta2[1:,:] = delta2[1:,:]  + (lamda * Theta2[1:,:])
    G = np.concatenate([delta1.ravel(), delta2.ravel()])
    '''
    kk = 0
    kk = kk+1
    print(kk)
    '''
    global KK
    KK = KK + 1
    print(KK)
    
    J = (-1/m) * J + (0.5*lamda/m) * (np.sum(np.square(Theta1[1:,:])) + np.sum(np.square(Theta2[1:,:])))    
    return J, (G/m).flatten() # dz2.shape, dz3.shape,delta2.shape,delta1.shape   # p, np.mean(p == y)

## If you want to check, run the CheckG function, it will give the 1st 10 terms. Then run D1 = NN(X, y, nn_params)[1]
## and then D1[1][0:10]

def CheckG(nn_params):
    eps = 1e-4
    grad = np.zeros(nn_params.shape)
    #New_nn_params = np.copy(nn_params)
    for i in range(10):                 # nn_params.shape[0]
        New_nn_params1 = np.copy(nn_params)
        New_nn_params2 = np.copy(nn_params)
        New_nn_params1[i] = New_nn_params1[i]- eps
        Ls1 = NN(X, y, New_nn_params1)[0]
        New_nn_params2[i] = New_nn_params2[i]+ eps # as if + eps
        Ls2 = NN(X, y, New_nn_params2)[0]
        grad[i] = (Ls2-Ls1)/(2*eps)
        print(grad[i])
    return grad

'''
def SGGrad(z):
    GZ = 1 / (1 + np.exp(-z))
    GZ_prime = np.multiply(GZ, (1 - GZ))
    return GZ_prime
'''

def IRandWeights(Lin, Lout): #Lin is the number of neurons in the input layer to the theta, similarlly Lout
    eps = 0.12
    #eps = np.sqrt(6) / (np.sqrt(Lin+Lout))
    Theta_rand = np.random.rand(Lout, 1 + Lin) * 2 * eps - eps  # +1 for the bias term in the input layer
    return Theta_rand

import scipy.optimize as op
Theta11 = IRandWeights(400, 25)
Theta22 = IRandWeights(25, 10)

initial_nn_params = np.concatenate([Theta11.ravel(),Theta22.ravel()])

# This function is needed so that the minimizing function would work. The reason is the optimization function takes your function and substitute x0 as initial guess. Over here the NN function takes 3 inputs which would not work.
# So to over come this, we made a mediator function that take on input and drop it in the NN function along with X and y and return the cost and gradiant
# you could watch this video to understand more how scipy works on simple examples: https://www.youtube.com/watch?v=G0yP_TM-oag
# Also read this question on stack overflow: https://stackoverflow.com/questions/49776015/scipy-minimize-typeerror-numpy-float64-object-is-not-callable-running
def reduced_cost_func(initial_nn_params):  
    
    return NN(X, y, initial_nn_params)


results = op.minimize(reduced_cost_func, 
                   initial_nn_params,
                   method="TNC",
                   jac=True,
                   options={'maxiter':100, "disp":True})

params = results.x


def predict(X, params):
    Theta1 = params[0:(25*401)].reshape(25,401) # (25,401)
    Theta2 = params[(25*401):((25*401)+(10*26))].reshape(10,26) # (10,26)
    z2 = np.dot(X,Theta1.T)
    A2 = SG(z2)
    A2 = np.hstack([np.ones((A2.shape[0],1)),A2])
    z3 = np.dot(A2,Theta2.T)
    A3 = SG(z3)
    p = np.zeros((X.shape[0],1))
    m = X.shape[0]
    for i in range(m):
        p[i] = list(A3[i]).index(np.amax(A3[i]))
    return np.mean(p == y)


