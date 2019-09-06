import tensorflow as tf
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

print("data has been loaded.")

x_image = X_test[0]
sess = tf.InteractiveSession()


# 建立一个tensorflow的会话

# 初始化权值向量
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 初始化偏置向量
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 二维卷积运算，步长为1，输出大小不变
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 池化运算，将卷积特征缩小为1/2
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 给x，y留出占位符，以便未来填充数据
x = tf.placeholder("float", [None, 2048])
#y_ = tf.placeholder("float", [None, 10])
    # 设置输入层的W和b
W = tf.Variable(tf.zeros([2048, 10]))
b = tf.Variable(tf.zeros([10]))
# 计算输出，采用的函数是softmax（输入的时候是one hot编码）
#y = tf.nn.softmax(tf.matmul(x, W) + b)

# 第一个卷积层，5x5的卷积核，输出向量是32维
w_conv1 = weight_variable([3, 3, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])
# 图片大小是28*28，,-1代表其他维数自适应
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
# 采用的最大池化，因为都是1和0，平均池化没有什么意义

# 第二层卷积层，输入向量是32维，输出64维，还是5x5的卷积核
w_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 第三层卷积层，输入向量是64维，输出128维，还是3x3的卷积核
w_conv3 = weight_variable([3, 3, 64,128])
b_conv3 = bias_variable([128])

h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

   # 全连接层的w和b
w_fc1 = weight_variable([2048, 2048])
b_fc1 = bias_variable([2048])
    # 此时输出的维数是256维
h_pool3_flat = tf.reshape(h_pool3, [-1, 2048])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1) + b_fc1)
    # h_fc1是提取出的256维特征，很关键。后面就是用这个输入到SVM中
    #比方说，我训练完数据了，那么想要提取出来全连接层的h_fc1，
    #那么使用的语句是sess.run(h_fc1, feed_dict={x: input_x})，返回的结果就是特征向量
x_test = x_image
x_temp = []
x_temp.append(sess.run(h_fc1, feed_dict={x: np.array(x_image).reshape((1, 2048))})[0])
print(x_temp)