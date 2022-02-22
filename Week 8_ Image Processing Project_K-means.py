import numpy as np
from scipy.io import loadmat
import math
from matplotlib.animation import FuncAnimation # for the animation task
import matplotlib.pyplot as pyplot
import matplotlib as mlp

data = loadmat(r'ex7data2') # add your path

# to know what's insdide data
for keys in data.keys():
    print(keys)

X = data['X']
r'''
## Constructing the K-means algorithm ##
centroids = kMeansInitCentroids(X,K) # this is the random initilization function that we will build later to initialize the position of the clusters' centroids
for i in range(iterations): # iterations will be a variable for us to set, it indicates how many times it should do the 2 main steps(assign Ci and update Uk) in the K-mean algorithm
    idx = findClosestCentroids(X,centroids) # idx is like C(i). In other words, idx represents to which a specific exmaple Xi belongs to a specific cluster k
    centroids = computeMeans(X,idx,K) # after the assigments of all Xi to C(i) (a cluster). Compute the mean of the cluster
'''
## Finding Closest Centroids ##

# vectorized version of this function #
def findClosestCentroids(X,centroids):
    K = centroids.shape[0] # centorids.shape or len(centroids)
    idx = np.zeros((X.shape[0],1))
    distanceBetSmaplesCentroids = np.zeros((X.shape[0],K))
    for j in range(K):
        distanceBetSmaplesCentroids[:,j]=np.sqrt(np.sum((X - centroids[j])**2,axis=1)) # Meaning distanceBetSmaplesCentroids[:,0] = distance between each sample in X and centroid[0]
    idx = np.argmin(distanceBetSmaplesCentroids,axis=1) # now find the shortest distance in each row, which crossponds to the shortest distance to the cluster k in column k
    sumOfShortestDists = np.sum(np.min(distanceBetSmaplesCentroids,axis=1))
    return idx.reshape(X.shape[0],1), distanceBetSmaplesCentroids, sumOfShortestDists # sumOfShortestDists could represent the cost function,

r'''
# Same function but using for loops, takes longer time to excute #
from scipy.spatial import distance # to calculate the distance between two points with any n-dimentions
def findClosestCentroids(X,centroids):
    K = centroids.shape[0] # centorids.shape or len(centroids)
    idx = np.zeros((X.shape[0],1))
    distanceBetSmaplesCentroids = np.zeros((1,K))
    # The idea here is to find the distance between each sample X[i] and all the given centroids one at a time and put these distances in a variable (distanceBetSmaplesCentroids).
    # Then find find the shortest distance in this variable and find the centroid number associated with it and put it in idx variable
    for i in range(X.shape[0]):
        for j in range(K):
            distanceBetSmaplesCentroids[0,j] = distance.euclidean(X[i],centroids[j]) # should return dist between [X[i] and U1, X[i] and U2 , X[i] and Uk]
            # distance.cdist() could be used between a whole array and a point, but it only work for 2 dimentional points. For more details about it check:
            # https://stackoverflow.com/questions/48609781/distance-between-one-point-and-rest-of-the-points-in-an-array/48610147
        # if you want to see while it's working you could uncomment the next few lines
        #print(distanceBetSmaplesCentroids)
        #print(np.min(distanceBetSmaplesCentroids))
        #print(np.argwhere(distanceBetSmaplesCentroids == np.min(distanceBetSmaplesCentroids))[0][1]) or the next line
        #print(np.argmin(distanceBetSmaplesCentroids))
        #idx[i] = np.argmin(distanceBetSmaplesCentroids)
        idx[i] = np.argwhere(distanceBetSmaplesCentroids == np.min(distanceBetSmaplesCentroids))[0][1] #np.argmin(distanceBetSmaplesCentroids) # this will capture the order/index of the shortest distance
        # or np.argwhere(distanceBetSmaplesCentroids == np.min(distanceBetSmaplesCentroids))[0][1]), this np.min(distanceBetSmaplesCentroids))[0][1] was used bec np.min return the [value, shape, datatype], and the value has rows and coulmns and we want only the column  so [0]then[1] was used
    return idx
'''

## Computing centroid means ##

# vectorized version of th function

def computeCentroids(X,idx,K):
    m, n = X.shape
    centroids = np.zeros((K,n))
    for i in range(K):
        Extract = np.where(idx == i)[0] # Extract the rows/samples that belongs to cluster idx == 0,1,..,K
        # print('Vectorized')
        # print (Extract)
        centroids[i] = np.mean(X[Extract],axis=0) # then find the mean of
        # print(centroids[i])
    #centroids[K] = np.zeros((1,K)) # in case you want one of the centroids/color to be black
    return centroids


# If you are using this function it will give the same results as the one above. However, if you are using a highly pixelated image with a lot of decimals, you might find a slight difference due to approximations, but wont lead to any signifcance difference in the image to be displayed
def computeCentroids1(X,idx,K):
    m, n = X.shape
    centroids = np.zeros((K,n))
    for i in range(K):
        for j in range(n):
            Extract = np.where(idx == i)[0] # Extract the rows/samples that belongs to cluster idx == 0,1,..,K
            # print('For LOOP:')
            # print (Extract)
            centroids[i,j] = np.mean(X[Extract][:,j]) # then find the mean of
        # print(centroids[i])
    return centroids

# the same function using for loops
r'''
def computeCentroids(X,idx,K):
    n = X.shape[1]
    U = np.zeros((K,n))
    Ck = np.zeros((K,1))
    for j in range(X.shape[0]):
        for i in range(K):
            if idx[j] == i:
                U[i] = U[i] + X[j]
                Ck[i] = Ck[i] + 1
    return U , Ck
'''

## Intializing the centriods of the clusters K ##

def kMeansInitCentroids(X, K):
    randidx = np.random.permutation(X.shape[0]) # pick a number from 0 - 300 (number of samples)
    # the reason we used this instead of randint, because randint draws "with replacement": some numbers can occur multiple times while others can be missing.
    centroids = X[randidx[:K],:] # X[randidx[:3],:]
    # after getting three random numbers from randidx, use them to pick the points from X
    return centroids

#initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

#idx = findClosestCentroids(X,initial_centroids)

#C = computeCentroids(X, idx, 3)

# put the image in an array
A = mlp.pyplot.imread(r'your_pic.PNG') # add your pic
#A = mlp.pyplot.imread(r'C:\Users\mahmo\OneDrive\Desktop\Data Science\ML\EX7\bird_small.png')
K = 6
# scale the features to be between 0-1
A = A / 255 # this feature scaling for the picture, since 0 is black and 255 is white, we could make all the colors between 0 and 1 by dividing by 255
X = A.reshape(-1,A.shape[-1]) 
J = [10000000000000] # I made it huge so that when you compare to it in the for loop, any other values could replace it
## putting the K-mean algorithm together ##
iterations = 5 # it depends how many iteration you want to go through
for i in range(iterations): # iterations will be a variable for us to set, it indicates how many times it should do the 2 main steps(assign Ci and update Uk) in the K-mean algorithm
    I_C = kMeansInitCentroids(X, K) # intialize the centroids
    # print(I_C)
    idx = findClosestCentroids(X,I_C)[0] # idx is like C(i). In other words, idx represents to which a specific exmaple Xi belongs to a specific cluster k
    #centroids1 = computeCentroids1(X,idx,K)
    J.append(findClosestCentroids(X,I_C)[2])
    #print(J)
    # only if the cost function is less than the one just computed, you compute the centroids, so that u don't waste processing power!
    if J[i+1] <= min(J): # now in the first iteration J[1] for sure <= minimum of J
        print(J[i+1])
        centroids = computeCentroids1(X,idx,K) # after the assigments of all Xi to C(i) (a cluster). Compute the mean of the cluster
    print(i)
    print('#################################################')
    #print(idx)
    #print(centroids)
## Mapping the picture with the new colors/centroids picked up from the previous section ##
X_recovered = np.zeros((X.shape))
#X_recovered1 = np.zeros((X.shape))
for i in range(idx.shape[0]):
    for j in range(K):
        if idx[i] == j:
            X_recovered[i] = centroids[j]
X_recovered = X_recovered.reshape(A.shape)

## plotting the original picture and reduced one ##
fig, ax = pyplot.subplots(1,2, figsize = (8,4))
ax[0].imshow((A*255))
#ax[0] = set_title('org')
#ax[0].grid(False)
ax[1].imshow((X_recovered*255))
#ax[1] = set_title('reduced')
#ax[1].grid(False)
pyplot.show()         
