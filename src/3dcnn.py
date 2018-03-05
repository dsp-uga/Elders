
from keras.datasets import shapes_3d

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.layers.convolutional import Convolution3D, MaxPooling3D

from keras.optimizers import Adam

from keras.utils import np_utils

import tensorflow

''' Arguments to be taken from users form command line Data Location CNN Training parameters all 9, Data Parameters all 3
 will have to have the arguments handling function here so that it will be easy for  users to test different loss functions etc 
'''

# Data Paremeters

test_split = 0.2   # How much split you want to be test data 0.2 would mean 20% test and 80% train
dataset_size = 5000 # Number of images this cant be hard coded will have to find by counting number of images in the folder
patch_size = 32

# CNN Training parameters
optimizer_method="Adam"
loss_method="categorical_crossentropy"
batch_size = 128
nb_classes = 2
nb_epoch = 10
nb_filters = [16, 32]
nb_pool = [3, 3]
nb_conv = [7, 3]
learning_rate=0.1
decay=0.00001 # How much the learning rate will decay as epochs goes on
momentum=0.9



# This fucntion will get the train/test split and convert into a 4D tensor that will we usefull for 3D CNN
(X_train, Y_train),(X_test, Y_test) = shapes_3d.load_data(test_split=test_split,dataset_size=dataset_size, patch_size=patch_size)

''' May have to delete 
# convert class vectors to binary class matrices

Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)
'''



# Creating the Model
def model_build_3dcnn():
    model = Sequential()
    model.add(Convolution3D(nb_filters[0],nb_depth=nb_conv[0], nb_row=nb_conv[0], nb_col=nb_conv[0], border_mode='valid',input_shape=(1, patch_size, patch_size, patch_size), activation='relu'))
    model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0])))
    model.add(Dropout(0.5))
    model.add(Convolution3D(nb_filters[1],nb_depth=nb_conv[1], nb_row=nb_conv[1], nb_col=nb_conv[1], border_mode='valid',activation='relu'))
    model.add(MaxPooling3D(pool_size=(nb_pool[1], nb_pool[1], nb_pool[1])))
    model.add(Flatten())
    model.add(Dropout(0.5))
    return model

# WIll have to add ROI here also This is segmentation and not classification so nb_classes is questionable
'''
model.add(Dense(16, init='normal', activation='relu'))

model.add(Dense(nb_classes, init='normal'))

model.add(Activation('sigmoid'))
'''

model=model_build_3dcnn()
if optimizer_method=="Adam":
    optimizer=Adam(lr=learning_rate, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss=loss_method, optimizer=optimizer)

# Here the model starts to train on the data as this is lazy execution , Basically the start button
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2,validation_data=(X_test, Y_test))


# Check the training against the test data provided
score = model.evaluate(X_test, Y_test, batch_size=batch_size, show_accuracy=True)

print('Test score:', score[0])

print('Test accuracy:', score[1])