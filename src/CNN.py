import tensorflow.keras as keras
from tensorflow.keras.layers import Activation, Dense, Conv2D, Input, Softmax
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, Flatten
import numpy as np
from tensorflow.keras import backend as K
import preprocessing
import utils

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
    X = Dropout(0.5)(X)

    X = Dense(3, activation='relu', name='fc1')(X)
    X = Dropout(0.5)(X)
    X = Softmax()(X)
    #Dense(units=no_classes, activation='softmax')(x)

    CNNmodel = keras.Model(inputs=X_input, outputs=X, name='CNNmodel')

    return CNNmodel

def trainCNN(model, ep):

    X_train, y_train = preprocessing.loadTrainingSet()
    X_train, y_train = preprocessing.extract_patches(X_train, y_train)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='SGD', metrics=["accuracy"])

    history = model.fit(X_train, y_train, epochs=ep, batch_size=32)
    model.save("CNN.h5")


def evaluateCNN(CNNmodel, y_test_imgs):

    X_test, y_test = preprocessing.loadTestSet()

    print("Evaluation patch-wise")
    preds = CNNmodel.evaluate(x=X_test, y= y_test)

    print("Test Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))

    y_pred = []
    yPredictedProbs = CNNmodel.predict(X_test)
    yPredicted = yPredictedProbs.argmax(axis=1)
    yPredicted = np.reshape(yPredicted, (yPredicted.shape/336, 336))

    for pred in yPredicted:
        count_bins = np.bincount(pred)
        y_pred.append(np.argmax(count_bins))
    utils.print_confusion_matrix(y_test_imgs, y_pred, 3, ['FL', 'MCL', 'CLL'])

    '''
    shape = X_test.shape[0]
    y_pred = []
    print("Evaluating...")
    for i in range(1, shape, 336):
        print((i / 336) / (shape / 336) * 100 + "% Completed")

        # Prediction of the patches in a single original image (336 patches per image)
        yPredictedProbs = CNNmodel.predict(X_test[i:i + 336, :, :, :])
        yMaxPredictedProbs = np.amax(yPredictedProbs, axis=1)
        yPredicted = yPredictedProbs.argmax(axis=1)
        count = np.bincount(yPredicted)
        y_pred.append(np.argmax(count))
    utils.print_confusion_matrix(y_test_imgs, y_pred, 3, ["CLL", "FL", "MCL"])
    '''



