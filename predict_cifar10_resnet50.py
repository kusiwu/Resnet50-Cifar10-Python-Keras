"""
Created on Wed Nov 21 01:09:08 2018

@author: kusiwu
@git: https://github.com/kusiwu
"""
# DEPENDENCIES
from keras import optimizers
from keras.datasets import cifar10
from tensorflow.python.client import device_lib
from keras.models import load_model
import scipy.io as sio
import numpy as np
from sklearn.utils import shuffle

img_width, img_height = 32, 32
dimensionx=3

nb_classes = 10


(X_train, y_train), (X_test, y_test) = cifar10.load_data()
#Load Training and Test data

print("shuffling test dataset randomly!")
# X_train, y_train = shuffle(X_train, y_train, random_state=0)
X_test, y_test = shuffle(X_test, y_test, random_state=1)


xTestPictures=X_test
yTestExpectedLabels=y_test
xTestPictures=X_test[0:1000,:,:,:]
yTestExpectedLabels=y_test[0:1000];

#xTestPictures=xTestPictures / 255.0;

modelpath='./trained_models/resnet50model1.h5'
model = load_model(modelpath)
print(modelpath + ' is loaded!')

# model.summary()
# print(device_lib.list_local_devices())

yFit = model.predict(xTestPictures, batch_size=10, verbose=1)
y_classes = yFit.argmax(axis=-1)
print("Found classes from prediction:");
print(y_classes.flatten())
np.savetxt('./logs/PredictedClasses.out', y_classes.flatten(), delimiter=',')   # X is an array


print("\n\nTrue classes:");
print(yTestExpectedLabels.flatten())
np.savetxt('./logs/TrueClasses.out', yTestExpectedLabels.flatten(), delimiter=',')   # X is an array


diffpredictionvstruth=yTestExpectedLabels.flatten()-y_classes.flatten()
print("\n\nDiff classes:");
print(diffpredictionvstruth)
np.savetxt('./logs/DiffClasses.out', diffpredictionvstruth, delimiter=',')   # X is an array


nb_validation_samples = xTestPictures.shape[0]
print("\n\n Wrong class number:");
print(str(len(np.nonzero(diffpredictionvstruth)[0]))+ '/' + str(nb_validation_samples) + ' is wrongly classified')

print("\n\n abs(dif error) summation:");
print(np.sum(abs(diffpredictionvstruth)))
#print()
#print(yFit)
del model