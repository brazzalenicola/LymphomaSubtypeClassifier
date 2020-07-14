import tensorflow.keras as keras
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Add
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras import backend as K
import preprocessing

# Implementation of the Improved Residual Recurrent Neural Network
# Convolution 2D with batch normalization
def conv_bn(X, nb_filter, num_row, num_col, padding='same', strides=(1, 1), use_bias=False):
    """
    Utility function to apply conv + BN.
    """
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    #num_row, num_col size of kernel
    X = Conv2D(nb_filter, (num_row, num_col), strides=strides, padding=padding, use_bias=use_bias,
               kernel_regularizer=keras.regularizers.l2(1e-5),
               kernel_initializer=keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal',
                                                                     seed=None))(X)
    X = BatchNormalization(axis=channel_axis, momentum=0.9997, scale=False)(X)
    X = Activation('relu')(X)
    return X


# Recurrent convolutional layer
def RCL(input, kernel_size, filedepth):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    X = Conv2D(filters=filedepth, kernel_size=kernel_size, strides=(1, 1), padding='same',
                   kernel_regularizer=keras.regularizers.l2(1e-4),
                   kernel_initializer=keras.initializers.VarianceScaling(scale=2.0, mode='fan_in',
                                                                         distribution='normal', seed=None))(input)

    path0 = BatchNormalization(axis=channel_axis, momentum=0.9997, scale=False)(X)
    path0 = Activation('relu')(path0)

    RCL = Conv2D(filters=filedepth, kernel_size=kernel_size, strides=(1, 1), padding='same',
                 kernel_regularizer=keras.regularizers.l2(0.00004),
                 kernel_initializer=keras.initializers.VarianceScaling(scale=2.0, mode='fan_in',
                                                                       distribution='normal',
                                                                       seed=None))

    conv1 = RCL(path0)
    stack3 = Add()([X, conv1])

    path1 = BatchNormalization(axis=channel_axis, momentum=0.9997, scale=False)(stack3)
    path1 = Activation('relu')(path1)

    conv3 = Conv2D(filters=filedepth, kernel_size=kernel_size, strides=(1, 1), padding='same',
                   weights=RCL.get_weights(),
                   kernel_regularizer=keras.regularizers.l2(0.00004),
                   kernel_initializer=keras.initializers.VarianceScaling(scale=2.0, mode='fan_in',
                                                                         distribution='normal', seed=None))(path1)
    stack5 = Add()([conv1, conv3])
    path2 = BatchNormalization(axis=channel_axis, momentum=0.9997, scale=False)(stack5)
    path2 = Activation('relu')(path2)

    conv4 = Conv2D(filters=filedepth, kernel_size=kernel_size, strides=(1, 1), padding='same',
                   weights=RCL.get_weights(),
                   kernel_regularizer=keras.regularizers.l2(0.00004),
                   kernel_initializer=keras.initializers.VarianceScaling(scale=2.0, mode='fan_in',
                                                                         distribution='normal', seed=None))(path2)
    stack7 = Add()([conv1, conv4])
    path3 = BatchNormalization(axis=channel_axis, momentum=0.9997, scale=False)(stack7)
    path3 = Activation('relu')(path3)

    return path3


def IRCNN_block(input):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    branch0 = RCL(input, (1, 1), 64)

    branch1 = RCL(input, (3, 3), 128)

    branch2 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    branch2 = RCL(branch2, (1, 1), 64)

    x = keras.layers.concatenate([branch0, branch1, branch2], axis=channel_axis)
    return x


def IRRCNN_model(input):
    if K.image_data_format() == 'channels_first':
        #     inputShape = (3, 64, 64)
        channel_axis = 1
    else:
        #     inputShape = (64, 64, 3)
        channel_axis = -1

    model = Conv2D(32, (3, 3), strides=(2, 2), padding='valid')(input)
    model = conv_bn(model, 32, 3, 3, padding='valid')
    model = conv_bn(model, 64, 3, 3)

    model = IRRCNN_block(input)

    model = conv_bn(model, 32, 3, 3, strides=(2, 2), padding='valid')
    model = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(model)
    model = Dropout(0.5)(model)

    model = IRRCNN_block(input)

    model = conv_bn(model, 32, 3, 3, strides=(2, 2), padding='valid')
    model = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(model)
    model = Dropout(0.5)(model)

    model = IRRCNN_block(input)

    model = conv_bn(model, 32, 3, 3, strides=(2, 2), padding='valid')
    model = GlobalAveragePooling2D()(model)
    model = Dropout(0.5)(model)

    return model


def IRRCNN_block(input):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    net = IRCNN_block(input)
    net1 = Conv2D(256, (1, 1), padding='valid')(input)
    net = Add()([net, net1])

    return net

def trainingIRRCNN(image_size, ep):
    X_train, y_train = preprocessing.loadTrainingSet()

    if K.image_data_format() == 'channels_first':
        inputs = keras.layers.Input(shape=(3, image_size, image_size))
    else:
        inputs = keras.layers.Input(shape=(image_size, image_size, 3))

    # MODEL
    x = keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='valid')(inputs)
    x = IRRCNN_model(x)
    x = keras.layers.Dense(units=3, activation='softmax')(x)

    model = keras.models.Model(input, x, name='IRRCNN')

    opt = keras.optimizers.SGD(lr=1e-2)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
    history = model.fit(x = X_train, y = y_train, epochs=ep, batch_size=32)
    model.save("IRRCNN.h5")