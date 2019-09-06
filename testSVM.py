# import SVMTestFile
import selfSVM
import h5py
import numpy as np
from time import time
from sklearn.decomposition import PCA
from scipy import stats
import pandas as pd

# import SVMforFashion
# SVMforFashion.testPoly('C:/Users/Derek/Desktop/comp5318Ass/data/train/images_training.h5','C:/Users/Derek/Desktop/comp5318Ass/data/train/labels_training.h5','C:/Users/Derek/Desktop/comp5318Ass/data/test/images_testing.h5','C:/Users/Derek/Desktop/comp5318Ass/data/test/labels_testing_2000.h5',1000,200)

# KNNforFashion.FashionClassTest()
# import SVMforFashion
# from numpy import *
#
# # SVMTestFile.testDigits(kTup = ('rbf', 10))
# SVMforFashion.testFashion(kTup = ('rbf', 10))
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

num_train = 20000
num_test = 2000
num_val = 1000
num_check = 100
#change data into two-dimensional
X_train = np.reshape(data,(-1,784))
X_test = np.reshape(data_predict,(-1,784))
#only first 2000 test labels are provided
Y_test = Y_test[:2000]
X_test = X_test[:2000]
X_train = X_train[:20000]
Y_train = Y_train[:20000]

X_val = X_train[-1000:]
Y_val = Y_train[-1000:]
#Introduce the exact label
type1 = {0:'T-shirt', 1:'Trouser', 2:'Pullover',3:'Dress', 4:'Coat', 5:'Sandal', 6:'Shirt',
        7:'Sneaker', 8:'Bag', 9:'Ankle boot'}


from sklearn import preprocessing
# 预处理: 标准化
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print('preprocessing has been done!')

t0=time()
# apply PCA
print('apply PCA')
n_components = 0.95
pca = PCA(n_components=n_components, svd_solver='full',
          whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

print("Projecting the input data on the eigenvalue orthonormal basis")
t0 = time()
X_train = pca.transform(X_train)
X_val = pca.transform(X_val)
X_test = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))

t0=time()
# 创建训练样本
x_tr = X_train[0:-1]
y_tr = Y_train[0:-1]

# 创建验证样本
x_val = X_train[num_train:(num_train + num_val)]
y_val = Y_train[num_train:(num_train + num_val)]

# 创建测试样本
x_te = X_test
y_te = Y_test

# 从训练样本中取出一个子集作为梯度检查的数据
mask = np.random.choice(num_train, num_check, replace=False)
x_check = x_tr[mask]
y_check = y_tr[mask]

# 计算训练样本中图片的均值
mean_img = np.mean(x_tr, axis=0)

# 所有数据都减去均值做预处理
x_tr += -mean_img
x_val += -mean_img
x_te += -mean_img
x_check += -mean_img
SVM = selfSVM.SVM()
SVM.train(X_train,Y_train,10 )
y_Pre = SVM.predict(x_te)
errorcount = 0

erro = []
erro_summary = []
erro_lable = []
erro_pre = []
for i in range(len(Y_test)):
    if(y_te[i] != y_Pre[i]):
        erro.append(i)
        erro.append(y_Pre[i])
        erro.append(y_te[i])
        erro_lable.append(y_te[i])
        erro_pre.append(y_Pre[i])
        erro_summary.append(erro_lable)
        print(i,y_Pre[i],y_te[i])
        errorcount+=1
        #print("Test number is ",str(i)," error count is ",errorcount," error rate is" ,errorcount/i)

print("Final error rate is %0.3f" ,(1- errorcount/len(Y_test)))
print("done in %0.3fs" % (time() - t0))
erro_summary = np.array(erro_summary)
#返回众数
#print(erro_summary)
erro_test = zip(erro_pre,erro_lable)

print(stats.mode(erro_pre)[0][0])
print(stats.mode(erro_lable)[0][0])
#gm = pd.Series(data=erro_summary)
print(erro_pre)
print(erro_lable)
result1 = pd.value_counts(erro_lable)
print (result1)
result2 = pd.value_counts(erro_pre)
print (result2)