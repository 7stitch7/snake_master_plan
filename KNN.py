import numpy as np
from numpy import *
import operator
import pandas as pd


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def FashionClassTest():
    data_train = pd.read_csv('/Users/fuqinwei/Desktop/assignment5318/fashionmnist/fashion-mnist_train.csv')
    csv = '/Users/fuqinwei/Desktop/assignment5318/fashionmnist/fashion-mnist_test.csv'
    data_test = pd.read_csv('%s' % csv)
    trainLabel = np.array(data_train.iloc[:, 0])
    m = len(trainLabel)
    trainMat = zeros((m, 784))
    for i in range(m):
        trainMat[i, :] = np.array(data_train.iloc[i, 1:])

    errorCount = 0.0
    TestLabel = np.array(data_test.iloc[:, 0])
    mTest = len(TestLabel)
    for i in range(mTest):
        vectorUnderTest = np.array(data_test.iloc[i, 1:])
        classifierResult = classify0(vectorUnderTest, trainMat, trainLabel, 3)
        classNumStr = TestLabel[i]
        print("the classifier came back with: %d, the real answer is: %d" %(classifierResult,classNumStr ))
        if(classifierResult != classNumStr):
            errorCount += 1.0

        print("\nthe test is: %d" % i)
        print("\nthe total number of errors is: %d" % errorCount)
        print("\nthe total error rate is: %f" % (errorCount/float(mTest)))

