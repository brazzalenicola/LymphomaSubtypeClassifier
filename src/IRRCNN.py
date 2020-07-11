import tensorflow.keras as keras
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Add
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from tensorflow.keras import backend as K


# Implementation of the Improved Residual Recurrent Neural Network
# Convolution 2D with batch normalization
def conv_bn(x, nb_filter, num_row, num_col, padding='same', strides=(1, 1), use_bias=False):
    """
    Utility function to apply conv + BN.
    """
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    x = Conv2D(nb_filter, (num_row, num_col), strides=strides, padding=padding, use_bias=use_bias,
               kernel_regularizer=keras.regularizers.l2(1e-5),
               kernel_initializer=keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal',
                                                                     seed=None))(x)

    x = BatchNormalization(axis=channel_axis, momentum=0.9997, scale=False)(x)
    x = Activation('relu')(x)
    return x


# Recurrent convolutional layer
def RCL(input, kernel_size, filedepth):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    conv1 = Conv2D(filters=filedepth, kernel_size=kernel_size, strides=(1, 1), padding='same',
                   kernel_regularizer=keras.regularizers.l2(1e-4),
                   kernel_initializer=keras.initializers.VarianceScaling(scale=2.0, mode='fan_in',
                                                                         distribution='normal', seed=None))(input)

    stack2 = BatchNormalization(axis=channel_axis, momentum=0.9997, scale=False)(conv1)
    stack2 = Activation('relu')(stack2)

    RCL = Conv2D(filters=filedepth, kernel_size=kernel_size, strides=(1, 1), padding='same',
                 kernel_regularizer=keras.regularizers.l2(0.00004),
                 kernel_initializer=keras.initializers.VarianceScaling(scale=2.0, mode='fan_in',
                                                                       distribution='normal',
                                                                       seed=None))

    conv2 = RCL(stack2)
    stack3 = Add()([conv1, conv2])
    stack4 = BatchNormalization(axis=channel_axis, momentum=0.9997, scale=False)(stack3)
    stack4 = Activation('relu')(stack4)

    conv3 = Conv2D(filters=filedepth, kernel_size=kernel_size, strides=(1, 1), padding='same',
                   weights=RCL.get_weights(),
                   kernel_regularizer=keras.regularizers.l2(0.00004),
                   kernel_initializer=keras.initializers.VarianceScaling(scale=2.0, mode='fan_in',
                                                                         distribution='normal', seed=None))(stack4)
    stack5 = Add()([conv1, conv3])
    stack6 = BatchNormalization(axis=channel_axis, momentum=0.9997, scale=False)(stack5)
    stack6 = Activation('relu')(stack6)

    conv4 = Conv2D(filters=filedepth, kernel_size=kernel_size, strides=(1, 1), padding='same',
                   weights=RCL.get_weights(),
                   kernel_regularizer=keras.regularizers.l2(0.00004),
                   kernel_initializer=keras.initializers.VarianceScaling(scale=2.0, mode='fan_in',
                                                                         distribution='normal', seed=None))(stack6)
    stack7 = Add()([conv1, conv4])
    stack8 = BatchNormalization(axis=channel_axis, momentum=0.9997, scale=False)(stack7)
    stack8 = Activation('relu')(stack8)

    return stack8


def IRCNN_block(input):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = RCL(input, (1, 1), 64)

    branch_1 = RCL(input, (3, 3), 128)

    branch_2 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    branch_2 = RCL(branch_2, (1, 1), 64)

    x = keras.layers.concatenate([branch_0, branch_1, branch_2], axis=channel_axis)
    return x


def IRRCNN_model(input):
    if K.image_data_format() == 'channels_first':
        #     inputShape = (3, 64, 64)
        channel_axis = 1
    else:
        #     inputShape = (64, 64, 3)
        channel_axis = -1

    net = Conv2D(32, (3, 3), strides=(2, 2), padding='valid')(input)
    net = conv_bn(net, 32, 3, 3, padding='valid')
    net = conv_bn(net, 64, 3, 3)

    net = IRRCNN_block(input)

    net = conv_bn(net, 32, 3, 3, strides=(2, 2), padding='valid')
    net = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(net)
    net = Dropout(0.5)(net)

    net = IRRCNN_block(input)

    net = conv_bn(net, 32, 3, 3, strides=(2, 2), padding='valid')
    net = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(net)
    net = Dropout(0.5)(net)

    net = IRRCNN_block(input)

    net = conv_bn(net, 32, 3, 3, strides=(2, 2), padding='valid')
    net = GlobalAveragePooling2D()(net)
    net = Dropout(0.5)(net)

    return net


def IRRCNN_block(input):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    net = IRCNN_block(input)
    net1 = Conv2D(256, (1, 1), padding='valid')(input)
    net = Add()([net, net1])

    return net


def print_confusion_matrix(true_labels, labels, num_classes, class_names):
    """
    Args:
        model: Keras model, already trained
        images: numpy tensor containing the test images
                [image_num, height, width, channels]
        labels: list of int, dataset labels (sparse representation)
        num_classes: int, number of classes
        class_names: list of string, name assiciated to each class
    Return:
        It prints the confusion matrix
    """
    # Get the predicted classifications for the test-set.
    predictions = labels
    # Get the true classifications for the test-set.

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=true_labels, y_pred=predictions)
    # Print the confusion matrix as text.
    # Plot the confusion matrix as an image.
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # Make various adjustments to the plot.
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.title('Confusion matrix')
    plt.show()
