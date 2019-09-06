from time import time
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


data_train = pd.read_csv('/Users/fuqinwei/Desktop/assignment5318/fashionmnist/fashion-mnist_train.csv')
data_test = pd.read_csv('/Users/fuqinwei/Desktop/assignment5318/fashionmnist/fashion-mnist_test.csv')
print("input data successful")
trainLabel = np.array(data_train.iloc[:10000, 0])
m = 10000
# m = len(trainLabel)
trainMat = zeros((m, 784))
for i in range(m):
    trainMat[i, :] = sign(np.array(data_train.iloc[i, 1:]))
TestLabel = np.array(data_test.iloc[:1000, 0])
mTest = 1000
# mTest = len(TestLabel)
TestMat = zeros((mTest, 784))
for i in range(mTest):
    TestMat[i, :] = sign(np.array(data_train.iloc[i, 1:]))
X_train = trainMat;
Y_train = trainLabel;
X_test = TestMat;
Y_test = TestLabel;
print("Set data successful")


n_components = 400
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

print("Projecting the input data on the eigenvalue orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))


print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                   param_grid, cv=5, iid=False)
clf = clf.fit(X_train_pca, Y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))
errorCount = 0
print(classification_report(Y_test, y_pred))
for i in range(len(Y_test)):
    if Y_test[i] != y_pred[i]:
        errorCount += 1
print("the training error rate is: %f" % (float(errorCount) / len(Y_test)))

print(confusion_matrix(Y_test, y_pred))

