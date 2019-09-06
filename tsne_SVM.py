from time import time
from sklearn.manifold import TSNE
#import matplotlib.pyplot as plt
from numpy import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import pandas as pd
import h5py
import seaborn as sns

#start = time.process_time()
#loading training and testing data
with h5py.File('/Users/fuqinwei/Desktop/assignment5318/data/train/images_training.h5','r') as H:
    data = np.copy(H['datatrain'])
    X_train = data.astype(np.float64)
with h5py.File('/Users/fuqinwei/Desktop/assignment5318/data/train/labels_training.h5','r') as H:
    Y_train = np.copy(H['labeltrain'])
with h5py.File('/Users/fuqinwei/Desktop/assignment5318/data/test/images_testing.h5','r') as H:
    data_predict = np.copy(H['datatest'])
    X_test = data_predict.astype(np.float64)
with h5py.File('/Users/fuqinwei/Desktop/assignment5318/data/test/labels_testing_2000.h5','r') as H:
    Y_test = np.copy(H['labeltest'])

#change data into two-dimensional
X_train = np.reshape(data,(-1,784))
X_test = np.reshape(data_predict,(-1,784))
#only first 2000 test labels are provided
Y_test = Y_test[:2000,]
X_test = X_test[:2000]
X_train = X_train[:10000]
Y_train = Y_train[:10000]
#Introduce the exact label
type1 = {0:'T-shirt', 1:'Trouser', 2:'Pullover',3:'Dress', 4:'Coat', 5:'Sandal', 6:'Shirt',
        7:'Sneaker', 8:'Bag', 9:'Ankle boot'}


from sklearn import preprocessing
# 预处理: 标准化
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print('preprocessing has been done!')



# apply PCA
print('apply PCA')
n_components = 500
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

print("Projecting the input data on the eigenvalue orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))


# apply SVM
print("Fitting the classifier to the training set")
from sklearn.svm import SVC
degree_choices = 2
best_score = 0.0
gamma_choices = [0.01,0.1,1,10]
C_choices = [0.01,0.1,1,10]
for degree in degree_choices:
    for gamma in gamma_choices:
        for C in C_choices:
            svm = SVC(C=C,kernel='poly',gamma=gamma,degree=degree)
            svm.fit(X_train_pca,Y_train)
            score = svm.score(X_test_pca,Y_test)
            print('C = %.3f,gamma = %.3f，degree = %d X_val accuracy = %.3f' % (C,gamma,degree,score))
            if score > best_score:
                best_score = score
                best_C = C
                best_gamma = gamma
                best_degree = degree

svm = SVC(C=best_C,kernel='poly',gamma=best_gamma,degree=best_degree) # 使用最佳参数，构建新的模型
svm.fit(X_train,Y_train) # 使用训练集进行训练
test_score = svm.score(X_test,Y_test) # 模型评估
print("Best score on validation set: %.3f" % (best_score))
print("Best parameters: C = %.3f, gamma = %.3f, degree = %d" % (best_C,best_gamma,best_degree))
print("Best score on test set: %.3f" % (test_score))