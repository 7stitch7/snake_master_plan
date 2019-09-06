import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.utils import to_categorical
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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

img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)
X_train = np.reshape(X_train,(-1,784))
X_test = np.reshape(X_test,(-1,784))
Y_test = Y_test[:2000]
X_test = X_test[:2000]
X_train = X_train[:20000]
Y_train = Y_train[:20000]
X_val = X_train[-1000:]
Y_val = Y_train[-1000:]
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
Y_val = to_categorical(Y_val)


#Introduce the exact label
type1 = {0:'T-shirt', 1:'Trouser', 2:'Pullover',3:'Dress', 4:'Coat', 5:'Sandal', 6:'Shirt',
        7:'Sneaker', 8:'Bag', 9:'Ankle boot'}


from sklearn import preprocessing
# 预处理: 标准化
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print('preprocessing has been done!')

X_train = np.reshape(X_train,(-1,28,28,1))
X_test = np.reshape(X_test,(-1,28,28,1))
X_val = np.reshape(X_val,(-1,28,28,1))

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

t0 = time()
batch_size = 256 #批量大小
num_classes = 10
epochs = 50

#input image dimensions
img_rows, img_cols = 28, 28
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 kernel_initializer='he_normal',
                 input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25)) #prevent overfitting
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax')) #Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer (only applicable if use_bias is True).

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


history = model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_val, Y_val))
score = model.evaluate(X_test, Y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

print("done in %0.3fs" % (time() - t0))