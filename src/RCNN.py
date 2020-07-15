import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
import preprocessing
import utils

def RCNN_model(input_shape):

    model = keras.applications.InceptionV3(include_top=False, weights=None, input_shape= input_shape, pooling='max')

    print(model.summary())

    new_model = Sequential()
    new_model.add(model)
    new_model.pop()
    print(model.summary())
    new_model.add(LSTM(128))
    new_model.add(LSTM(128))
    new_model.add(Dropout(0.5))
    new_model.add(Dense(3, activation='softmax'))

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=2100000, decay_rate=0.9)
    opt = keras.optimizers.SGD(learning_rate=lr_schedule, name="SGD")
    new_model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

    summary = new_model.summary()
    print(summary)

    return new_model

def trainRCNN(model, ep):

    X_train, y_train = preprocessing.loadTrainingSet()
    X_train, y_train = preprocessing.extract_patches(X_train, y_train)

    X_train = keras.layers.ZeroPadding2D(padding=11, data_format='channels_last')(X_train)
    history = model.fit(X_train, y_train, epochs=ep, batch_size=1)
    utils.plot_Accuracy_Loss(history)
    model.save("RCNN.h5")