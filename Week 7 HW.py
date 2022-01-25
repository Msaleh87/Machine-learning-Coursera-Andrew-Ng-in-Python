import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


#path = r'C:\Users\mahmo\OneDrive\Desktop\Data Science\ML\EX6\ex6data1.mat'
'''
data = loadmat(path)

for i in data.keys():
    print(i)

X , y = data['X'], data['y']

one = np.where(y==1)[0]
zero = np.where(y==0)[0]


plt.plot(X[one,0],X[one,1], 'bo')
plt.plot(X[zero,0],X[zero,1], 'ro')
plt.show()
'''

from sklearn import svm
# import seaborn as sns; sns.set(font_scale=1.2)
'''
model = svm.SVC(kernel='linear',C=1)
model.fit(X,y.ravel())
w = model.coef_[0]
b = model.intercept_[0]
a = -w[0]/w[1]
xx = np.linspace(-2,5)
yy = a * xx - (model.intercept_[0])/w[1]
plt.plot(xx,yy, linewidth=2,color='black')
plt.plot((-b/w[0],0),(0,-b/w[1]),color ='red') # python plots (X_values,Y_values), for example point1 = [1,2], ponit2 = [3,4], X_values= [point1[0], point2[0]]. Y_values = [point1[1],point2[1]]
#plt.plot((0,-b/w[0]),(-b/w[1],0),color ='red')  #correct one
#plt.plot((b/w[0],0) , (0,b/w[1]), color = 'red')
plt.plot(X[one,0],X[one,1], 'bo')
plt.plot(X[zero,0],X[zero,1], 'ro')
plt.show()
'''

def gaussianKernel(x1,x2,sigma):
    sim = np.exp(-sum((x1-x2)**2)/(2*sigma**2))
    #sim=np.exp(-np.power(x1-x2, 2).sum()/(2*sigma**2))
    return sim

#path = r'C:\Users\mahmo\OneDrive\Desktop\Data Science\ML\EX6\ex6data2'
'''
data2 = loadmat(path)

for i in data2.keys():
    print(i)
X , y = data2['X'], data2['y']

one = np.where(y==1)[0]
zero = np.where(y==0)[0]

plt.plot(X[one,0],X[one,1],'bo')
plt.plot(X[zero,0],X[zero,1],'r*')
plt.show()

from mlxtend.plotting import plot_decision_regions
model = svm.SVC(kernel='rbf',C=1,gamma=100)
model.fit(X,y.ravel())
plot_decision_regions(X=X, 
                      y=y.ravel(),
                      clf=model, 
                      legend=2)
plt.show()
'''
path = r'C:\Users\mahmo\OneDrive\Desktop\Data Science\ML\EX6\ex6data3'
data3 = loadmat(path)

for i in data3.keys():
    print(i)

X , y , Xval, yval = data3['X'] , data3['y'] , data3['Xval'] , data3['yval']

pos = np.where(y==1)[0]
neg = np.where(y==0)[0]
'''
plt.plot(X[pos,0],X[pos,1],'ro')
plt.plot(X[neg,0],X[neg,1],'b*')
plt.show()
'''
from mlxtend.plotting import plot_decision_regions
'''
model = svm.SVC(kernel='rbf',C=1,gamma =100)
model.fit(X,y.ravel())
H = model.predict(Xval)
#from mixtend.plotting import plot_decision_regions
plot_decision_regions(X=X,
                      y=y.reshape(X.shape[0],),
                      clf=model,
                      legend=1,
                      zoom_factor= 10)
#plt.yticks(np.arange(-0.5, 1))
#plt.xticks(np.arange(-0.5,0.3))
plt.show()
'''
'''
from sklearn.metrics import accuracy_score
C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
gamma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
AC = []
for i in C:
    for j in gamma:
        model = svm.SVC(kernel='rbf',C=i,gamma =j)
        model.fit(X,y.ravel())
        H = model.predict(Xval)
        AC.append([accuracy_score(H, yval),i,j])
'''
#model = svm.SVC(C=3,gamma=30,kernel='rbf')
model = svm.SVC(C=30,gamma=10,kernel='rbf')
model.fit(X,y.ravel())
plot_decision_regions(X=X,
                      y=y.reshape(X.shape[0],),
                      clf=model,
                      legend=1,
                      zoom_factor= 10)
plt.show()
'''

print(accuracy_score(H, yval))
print(np.mean(H==yval.ravel()))
'''
