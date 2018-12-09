#developed by kusiwu: 12.08.2018
#git:   https://github.com/kusiwu/

#import numpy as np
import os,sys
import time
from resnet50_32x32 import ResNet50
#from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten
from keras import optimizers
from keras.layers import Input
from keras.engine import Model
from keras.models import load_model
from keras.utils import np_utils
#from sklearn.utils import shuffle
#from sklearn.model_selection import train_test_split
from keras import callbacks
#from keras.utils.vis_utils import plot_model #for graphical demonstration of Network model #requires graphwiz. Not active for now...
from datetime import datetime
from keras.datasets import cifar10


WANNAFASTTRAINING=0
img_width, img_height = 32, 32
batch_trainsize=32 #decrease if you machine has low gpu or RAM
batch_testsize=32 #otherwise your code will crash.
nb_epoch = 5

#SGD: Gradient Descent with Momentum and Adaptive Learning Rate
#for more, see here: https://wiseodd.github.io/techblog/2016/06/22/nn-optimization/
learningrate=1e-3 #be careful about this parameter. 1e-3 to 1e-8 will train better while learningrate decreases.
momentum=0.8
num_classes = 10


(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)



if WANNAFASTTRAINING == 1 :
    X_train= X_train[0000:1500,:,:,:]
    y_train= y_train[0000:1500]
    X_test= X_test[000:1000,:,:,:]
    y_test= y_test[000:1000]


num_of_samples = X_train.shape[0]
###########################################################################################################################
# Custom_resnet_model_1
#Training the classifier alone
image_input = Input(shape=(img_width, img_height, 3))

previouslytrainedModelpath ='./trained_models/resneta50model1.h5'
if os.path.isfile(previouslytrainedModelpath):
    print('Loading previously trained model1...')
    model = load_model(previouslytrainedModelpath)
    print(previouslytrainedModelpath + ' successfully loaded!')
    custom_resnet_model=model
else :
    print('Initializing resnet50 model1')
    model = ResNet50(input_tensor=image_input, include_top=True,weights='imagenet')
    model.summary()
#    sys.exit(1)
    
    x = model.get_layer('res5a_branch2a').input
    x = GlobalAveragePooling2D(name='avg_pool')(x)
#    x = Dense(512, activation='relu',name='fc-1')(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation='softmax', name='output_layer')(x)
    custom_resnet_model = Model(inputs=image_input,outputs= out)


#custom_resnet_model.summary()

for layer in custom_resnet_model.layers[:]:
	layer.trainable = True

#custom_resnet_model.layers[-1].trainable

#custom_resnet_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
custom_resnet_model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=learningrate, momentum=momentum),
              metrics=['accuracy'])

custom_resnet_model.summary()


###### please install pydot with pip install pydot and download graphwiz from website :https://graphviz.gitlab.io/_pages/Download/Download_windows.html
####add graphwiz path to visualize model graph. No need for now.
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
#plot_model(custom_resnet_model, to_file='outputs/model1_plot.png', show_shapes=True, show_layer_names=True)


# callback for tensorboard integration
tb = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

# checkpoints. Save model if val_accuracy increases.
filepath="./trained_models/model1_-{epoch:02d}-{val_acc:.2f}_"
checkpoint = callbacks.ModelCheckpoint(filepath+f'{datetime.now():%Y-%m-%d_%H.%M.%S}'+'.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max',period=1)

t=time.time()
hist = custom_resnet_model.fit(X_train, y_train, batch_size=batch_trainsize, epochs=nb_epoch, verbose=1, validation_data=(X_test, y_test),callbacks=[tb,checkpoint])
print('Training time: %s' % (time.time()-t))
(loss, accuracy) = custom_resnet_model.evaluate(X_test, y_test, batch_size=batch_testsize, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

# serialize model to JSON
model_json = custom_resnet_model.to_json()
with open("./outputs/custom_resnet_model1.json", "w") as json_file:
    json_file.write(model_json)

#Save model
custom_resnet_model.save('./trained_models/resnet50model1.h5')
print('model1 resaved.')
del custom_resnet_model #prevent memory leak
sys.exit(1)
###############################################################################

###############################################################################
import matplotlib.pyplot as plt
# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(12)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])