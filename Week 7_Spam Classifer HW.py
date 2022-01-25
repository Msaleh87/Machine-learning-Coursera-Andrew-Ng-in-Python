import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import re
import time

# t = time.time()

vocablist = np.genfromtxt(r'C:\Users\mahmo\OneDrive\Desktop\Data Science\ML\EX6\vocab.txt', dtype=object)
# np.genfromtxt, it loops through each line of the text file and store the data in the line according to dtype. default value of delimiter = None, meaning:
# that the line is split along white spaces (including tabs) and that consecutive white spaces are considered as a single white space.
# to understand better why we used dtype = 'object' check this link:
# https://stackoverflow.com/questions/29877508/what-does-dtype-object-mean-while-creating-a-numpy-array

X = list(vocablist[:,1].astype(str))
X_indcies = list(vocablist[:,0].astype(int))
# .astype to convert the data from bytes to what is needed (string or integer)

# elapsed = time.time() - t

file = open(r'C:\Users\mahmo\OneDrive\Desktop\Data Science\ML\EX6\emailSample1.txt','r') # open this text file in the read mode
data = file.read() # if you put it inside the function ppEmail(), the function will run once fine. Then when it tries to read the file, it won't and this will mess up the function. Give it a try
def ppEmail(data):
    #print(data)
    hdrstart = data.find(chr(10) + chr(10)) # chr(10) means \n. If you check the text from public corpus for the last optional assignment, you will find that the body of the email starts after two enters, which is chr(10) + chr(10). This line will find where this is in the email
    # for example in eas_ham folder, the first email, data.find(chr(10) + chr(10)) = 
    data = data[hdrstart:]
    data = data.lower() # to make it all lower case so that the ML wouldn't treat them differently.
    data = re.compile('<[^<>]+>').sub(' ',data) # it means replace any thing between < and >.
    # [^<>]+, the carot at the begining ^ negates, meaning inverts its meaning, making it match anything but the characters in it, any character but <>.
    # the + sign for any combination of characters between the < and >
    # so anything between < > will be replaced by a space
    # re. sub() function is used to replace occurrences of a particular sub-string with another sub-string
    data = re.compile('[0-9]+').sub('number',data)
    # [0-9]+ means any character from 0 to 9 and in between and the plus sign for any combination
    data = re.compile('(http|https)://[^\s]*').sub('httpaddr',data) # this means (http or https) followed by :// followed by anything by white spaces [^\s] and it's repetion *
    # \s atches Unicode whitespace characters (which includes [ \t\n\r\f\v], and also many other characters
    # the asterisk is a metacharacter for zero or more instances of the preceding character.
    # The square brackets ([]) indicate a character class. A character class basically means that you want to match anything in the class, at that position, one time. [abc] will match the strings a, b, and c. In this case, your character class is negated using the caret (^) at the beginning - this inverts its meaning, making it match anything but the characters in it.
    # better explained at https://stackoverflow.com/questions/39402495/what-does-the-regex-s-mean/39402538
    data = re.compile('[^\s]+@[^\s]+').sub('emailaddr',data)
    # It means anything but white space (letters and numbers) and any combinations @ the same thing
    data = re.compile('[$]+').sub('dollar',data)
    # To match a literal '$' we enclose it in []

    data = re.split('[ @$/#.-:&*+=\[\]?!(){},''">_<;%\n\r]',data)
    data = [i for i in data if len(i) > 0] # called "list comperhension" # This line check word for word and put it in the list only if it's length is greater than 0 (not empty)
    #or data = list(filter(None,data))
    #print(data)
    word_indices = []

    import nltk # the library for stemming words, it has different functions inside it that do different types of stemming, one of them is PorterStemmer()
    # for more info you could check: https://stackoverflow.com/questions/24647400/what-is-the-best-stemming-method-in-python
    stemmer = nltk.PorterStemmer()
    processed_email = []
    for word in data: # go through each word to process it for the final time
        word = re.compile('[^a-zA-Z0-9]').sub('',word).strip() # [^a-zA-Z0-9] remove anything in the word but numbers and letters
        word = stemmer.stem(word) # stemming the word
        processed_email.append(word) # add each processed word to a new variable called processed_email
        if word in X:
            word_indices.append(X.index(word)+1) # I added +1 so that we start counting from 1 not zero, the numbers in word_indices will match the number in the pdf page 12
    return processed_email, word_indices
    
# print(' '.join(processed_email))
# print(word_indices)
## 2.2 Extracting Features from Emails ##
def Feature(data):
    
    PP = ppEmail(data)
    processed_email = PP[0]
    word_indices = PP[1]
    
    feature = []
    for word in X:
        if word in processed_email:
            feature.append(1)
        else: feature.append(0)
    feature.count(1) # counting the non zero elements = 44
    return feature


## 2.3 SVM classifier ##
from scipy.io import loadmat
path = r'C:\Users\mahmo\OneDrive\Desktop\Data Science\ML\EX6\spamTrain.mat'
data2 = loadmat(path)

for key in data2.keys():
    print(key)
Xtrain , ytrain = data2['X'], data2['y']
model = svm.SVC(C=0.1, kernel='linear')
model.fit(Xtrain,ytrain.ravel())
H = model.predict(Xtrain)
from sklearn.metrics import accuracy_score
print(accuracy_score(H,ytrain)) # or print(np.mean(H == ytrain.ravel())) # about 99.825 on the training set

path2 = r'C:\Users\mahmo\OneDrive\Desktop\Data Science\ML\EX6\spamTest.mat'
dataTest = loadmat(path2)
for key in dataTest.keys():
    print(key)
XTest, yTest = dataTest['Xtest'] , dataTest['ytest']
Test = model.predict(XTest)
print(accuracy_score(Test,yTest)) # 98.9%


## Check out the word that indicates that there is a scam ##
W = (model.coef_).T # Transpose to have it as column
index_maxCoef = np.where(W == np.max(W))[0][0]
# np.max(W) is 0.5, since the higest coefficient is 0.5 then let's find coefs greater than 0.45 and take the index to figure out what is the crossponding words
np.where(W > 0.45)[0] # the indicies are 297 and 1190 if X[297] = click.
#Pretty sure that this most likely indicate scam if they ask you to click. Try np.where(W > 0.3) you will dinf that the dollar as well is an indicator

print('#####################################################')
Email = open(r'C:\Users\mahmo\OneDrive\Desktop\Data Science\ML\EX6\emailSample2.txt','r') # Try putting the path of the email you want to check
Email_check = Email.read()
X_feature = Feature(Email_check)
print(' If the prediction is 1 then its spam, if it is not')
print(model.predict(np.array([X_feature])))

r'''
############################################
## How to read text files in a folder ##
import os
path = r'C:\Users\mahmo\OneDrive\Desktop\Data Science\ML\EX6\Email Samples\easy_ham'
os.chdir(path)
def RTF(file_path):
    with open(file_path, 'r') as f:
        print(f.read())

for file in os.listdir():
    file_path = f"{path}\{file}"
    RTF(file_path)
'''


import os
#############################################
## How to convert all the text file in a folder from spamassassin public corpus website into an X feature that we developed earlier ##
r'''
path = r'C:\Users\mahmo\OneDrive\Desktop\Data Science\ML\EX6\Email Samples\easy_ham'
# Pick the directory you want
os.chdir(path)


def RTF(file_path):
    with open(file_path, 'r') as f:
        XX = f.read()
        XX = np.array(Feature(XX))
        XX = XX.reshape(1,XX.shape[0])
    return XX
        
Accmulative_Features = np.zeros((1,len(X)))
for file in os.listdir():
    file_path = os.path.join(path,file) # f"{path}\{file}"
    Accmulative_Features = np.vstack((Accmulative_Features,RTF(file_path)))
'''

#############################################
## How to read all the text files in the subfolders['easy_ham', 'easy_ham_2', 'hard_ham', 'spam', 'spam_2'] without putting them in one folder manually ##

path_main = r'C:\Users\mahmo\OneDrive\Desktop\Data Science\ML\EX6\Email Samples'
os.chdir(path_main) # change the directory to path_main
MP = os.listdir()  # see what's inside this directory: ['easy_ham', 'easy_ham_2', 'hard_ham', 'spam', 'spam_2']


def RTF(file_path):
    Accmulative_Features = np.zeros((1,len(X))) # just to initiate the array and use vstack to add to it, later we will delete this line
    for path in MP: # ['easy_ham', 'easy_ham_2', 'hard_ham', 'spam', 'spam_2']
        # print('###########\n##############\n##############\n###########\n##############\n##############\n###########\n##############\n##############\n') # just a seperator to see things better in the consule
        M = os.path.join(path_main, path) # path_main\easy_ham then path_main\easy_ham_2 and so on
        os.chdir(M) # change directory to the subfolder now
        for file_path in os.listdir(): # Loop over all the files in the specified subfolder
            print(os.path.join(M,file_path))  # Just to see the progress as it happens and see the for loop happening
            F = os.path.join(path_main,path,file_path) # the path of the file
            with open(F, 'r',errors='ignore') as f:     # The reason I added errors = 'ignore'. inside spam folder there is a file 0123 which has latin character that the python gives error cuz it can't read it and stop the code
                XX = f.read()
                XX = np.array(Feature(XX))
                XX = XX.reshape(1,XX.shape[0])
                Accmulative_Features = np.vstack((Accmulative_Features,XX))
                print(Accmulative_Features.shape)
    return Accmulative_Features
ACC_Features = RTF(path_main)
Output_notSpam = np.zeros((2551+1400+250,1)) # these are the counts of files inside easy_ham, easy_ham_2, hard_ham.
# You could find the number of files in each folder using os.chdir(to the path you want to count), then len(os.listdir())
Output_Spam = np.ones((501+1396,1)) # spam has 501 files, spam_2 has 1396
y = np.vstack((Output_notSpam,Output_Spam))
X_Emails = ACC_Features[1:,:] # we started from one to delete the first row we created in the function
model = svm.SVC(C=0.1, kernel='linear')
model.fit(X_Emails,y.ravel())
Predict = model.predict(X_Emails)
Predict_Test = model.predict(XTest)
print(accuracy_score(Predict,y))
print('#########')
print(accuracy_score(Predict_Test,yTest))
# TEST Tomorrow to see if its working with SVM correctly!



