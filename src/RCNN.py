import tensorflow.keras as keras
from tensorflow.keras.layers import Activation, Dense, Conv2D, Input, Reshape, Add, LSTM
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, Flatten, concatenate
import preprocessing
import utils
import numpy as np

def RCNN_model(input_shape):

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
    last_cnn = Flatten()(X)

    X = Reshape((64 * 64, 3))(X_input)
    X = LSTM(32, name='lstm0', return_sequences=True)(X)
    rec_last = LSTM(32, name='lstm1')(X)

    #X = Add()([rec_last, last_cnn])
    X = concatenate([rec_last, last_cnn])

    #X = Dense(512, activation='relu', name='fc0')(X)
    #X = Dropout(0.5)(X)

    X = Dense(64, activation='relu', name='fc1')(X)
    X = Dropout(0.5)(X)

    X = Dense(3, activation='softmax')(X)

    rec_model = keras.Model(inputs=X_input, outputs=X, name='RCNNmodel')

    summary = rec_model.summary()
    print(summary)

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=54000, decay_rate=0.9)
    opt = keras.optimizers.SGD(learning_rate=lr_schedule, name="SGD")
    rec_model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

    return rec_model


def trainRCNN(rec_model, ep):
    X_train, y_train = preprocessing.loadTrainingSet()
    X_train, y_train = preprocessing.extract_patches(X_train, y_train, 3)

    # X_train = keras.layers.ZeroPadding2D(padding=11, data_format='channels_last')(X_train)
    history = rec_model.fit(X_train, y_train, epochs=ep, batch_size=32)
    utils.plot_Accuracy_Loss(history)
    rec_model.save("RCNN.h5")

def evaluateRCNN(RCNNmodel):

    X_test, y_test = preprocessing.loadTestSet()
    y_test_imgs = y_test
    X_test, y_test = preprocessing.extract_patches(X_test, y_test)


    print("Evaluation patch-wise:")
    preds = RCNNmodel.evaluate(x=X_test, y= y_test)

    print("Test Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))

    y_pred = []
    yPredictedProbs = RCNNmodel.predict(X_test)
    yPredicted = yPredictedProbs.argmax(axis=1)
    yPredicted = np.reshape(yPredicted, (yPredicted.shape/336, 336))

    for pred in yPredicted:
        count_bins = np.bincount(pred)
        y_pred.append(np.argmax(count_bins))

    print("Accuracy image wise: " + str((np.sum(y_test_imgs == y_pred) / 124) * 100) + " %")
    #utils.print_confusion_matrix(y_test_imgs, y_pred, 3, ['FL', 'MCL', 'CLL'])


