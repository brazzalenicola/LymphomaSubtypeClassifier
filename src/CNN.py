import tensorflow.keras as keras
from tensorflow.keras.layers import Activation, Dense, Conv2D, Input, Softmax
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, Flatten
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from tensorflow.keras import backend as K

def CNN_model(input_shape):
    if K.image_data_format() == 'channels_first':
        #     inputShape = (3, 64, 64)
        channel_axis = 1
    else:
        #     inputShape = (64, 64, 3)
        channel_axis = -1
    X_input = Input(input_shape)
    X = Conv2D(32, (5, 5), strides=(1, 1), name='conv0')(X_input)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='max_pool0')(X)
    X = Activation('relu')(X)

    X = Conv2D(32, (5, 5), strides=(1, 1), name='conv1')(X)
    X = Activation('relu')(X)

    X = AveragePooling2D(pool_size=(2, 2), strides=None, name='avg_pool0')(X)
    X = Conv2D(64, (5, 5), strides=(1, 1), name='conv2')(X)
    X = Activation('relu')(X)

    X = AveragePooling2D(pool_size=(2, 2), strides=None, name='avg_pool1')(X)
    X = Flatten()(X)
    X = Dense(64, activation='relu', name='fc0')(X)
    X = Dropout(0.33)(X)

    X = Dense(3, activation='relu', name='fc1')(X)
    X = Dropout(0.33)(X)
    X = Softmax()(X)
    #Dense(units=no_classes, activation='softmax')(x)

    CNNmodel = keras.Model(inputs=X_input, outputs=X, name='CNNmodel')

    return CNNmodel

