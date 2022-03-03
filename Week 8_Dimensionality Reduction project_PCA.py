r'''
Sources to get to understand PCA more:
First try to understand matrices rotation and Eigen vectors and values animated:
Change of basis:
https://www.youtube.com/watch?v=P2LTAUO1TdA
Eigen vectors and values:
https://www.youtube.com/watch?v=PFDu9oVAE-g

Then PCA animated and application:
https://www.youtube.com/watch?v=FgakZw6K1QQ
Then this article will simplify the math behind:
https://towardsdatascience.com/the-mathematics-behind-principal-component-analysis-fff2d7f4b643
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

data = loadmat(r'USE YOUR PATH\ex7data1')

for keys in data.keys():
    print(keys)

X = data['X']
#plt.plot(X[:,0],X[:,1],'x')
#plt.show()

def pca(X):
    m = X.shape[0]
    Sigma = (1/m)*np.dot(X.T,X)
    U, S = np.linalg.svd(Sigma)[:2]
    return U, S

def FeatureNormalization(X):
    mu = np.mean(X,axis=0) # Calculating the mean of each column/feature
    X_norm = X - mu
    std = np.std(X_norm,axis=0,ddof = 1)
    # std = sqrt(x-u/n), if you are using a sample of the population and the mean of that sample, then the common practice in statsitics is to divide by N-1. ddof means divisor by (N-ddof), which is N-1 in our case
    # more explanition could be found at Khan academy: https://www.khanacademy.org/math/ap-statistics/summarizing-quantitative-data-ap/more-standard-deviation/v/review-and-intuition-why-we-divide-by-n-1-for-the-unbiased-sample-variance
    X_norm = X_norm/std
    return X_norm, mu, std

X_norm = FeatureNormalization(X)[0]
U, S = pca(X_norm)
mu = FeatureNormalization(X)[1]

r'''
fig, ax = plt.subplots()
ax.plot(X[:, 0], X[:, 1], 'bo')

for i in range(2):
    ax.arrow(mu[0], mu[1], 1.5 * S[i]*U[0, i], 1.5 * S[i]*U[1, i],
             head_width=0.1)

# plt.show()
'''

def projectData(X,U,K):
    Z = np.dot(X,U[:,:K]) # X(mxn) x U[:,:K](nxk)
    return Z

def recoverData(Z,U,K):
    X_rec = np.dot(Z,U[:,:K].T) # U[:,:K](nxK) Z(mxk)
    return X_rec

K = 1
Z = projectData(X_norm,U,K)
X_rec = recoverData(Z,U,K)

fig , ax = plt.subplots(figsize=(5,5))
ax.plot(X_norm[:,0],X_norm[:,1], 'bo')
ax.plot(X_rec[:,0],X_rec[:,1], 'ro')

for xnorm, xrec in zip(X_norm,X_rec): # zip is a funtions that zips/shove the given variables in an object
    ax.plot([xnorm[0],xrec[0]],[xnorm[1],xrec[1]],'--k') # it draws a line between (x1,x2) (y1,y2)
plt.show()
## Loading the face data ##
data = loadmat(r'USE YOUR PATH\ex7faces')
Pics = data['X']

def DrawPics(X):
    m , n = X.shape
    # setting the dimension of each pic
    Ex_width = int(np.round(np.sqrt(n))) # each row is a picture of 1024, sqrt(1024) = 32
    Ex_Height = int(n/Ex_width) # Then the height would 32
    # Then you would have each row (i.e. picture) as 32 by 32 
    # setting up the frame, how many pics in rows and coloumns
    Display_r = int(np.floor(np.sqrt(m))) 
    Display_c = int(np.ceil(m/Display_r))
    # plotting commands
    fig, ax_arr = plt.subplots(Display_r ,Display_c , figsize = (6,6))
    fig.subplots_adjust(wspace=0.025,hspace=0.025)
    ax_arr = [ax_arr] if m == 1 else ax_arr.ravel() # to put all the subplots ax_arr in a vector array
    
    for i, ax in enumerate(ax_arr):
        ax.imshow(X[i].reshape(Ex_Height,Ex_width, order ='F'), cmap = 'gray') # order is fortrain for the pics
        ax.axis('off') # so that you don't have axis with numbers, which you don't need for pics


# DrawPics(Pics[:100,:]) # picked the first 100 images. If you put DrawPics(Pics[:144,:]), you will get 12 by 12 images

## 2.4.1 PCA on Faces ##
Pics_norm, mu, sigma = FeatureNormalization(Pics)
U , S = pca(Pics_norm)
print(U.shape) # 1024x1024
DrawPics(U[:, :100].T) # This like setting K = 36, because it's selecting all rows of U and the first 36 columns. Then U[:, :36].T is 36x1024
plt.gcf().suptitle('Reduced')
## 2.4.2 Dimensionality Reduction ##
K = 100
Z = projectData(Pics_norm,U,K)
print(Z.shape)
X_rec = recoverData(Z,U,K)
DrawPics(Pics[:100,:])
plt.gcf().suptitle('Original')
DrawPics(X_rec[:100,:])
plt.gcf().suptitle('Recovered')
plt.show()

## part 2.5 is in the K-mean file
