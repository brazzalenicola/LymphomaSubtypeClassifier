import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.layers import GlobalMaxPool2D, Input, Reshape, Add
import preprocessing
import utils
import CNN


def RCNN_model(input_shape):

    cnn_model = CNN.CNN_model(input_shape, class_layers = False)
    cnn_model.layers.pop()
    #print(cnn_model.summary())

    X_input = Input(input_shape)
    X = Reshape((64*64, 3))(X_input)
    X = LSTM(128, name='lstm0', return_sequences=True)(X)
    rec_last = LSTM(128, name='lstm1')(X)

    last_cnn = cnn_model.layers[-1]

    X = Add()([rec_last, last_cnn])

    X = Dense(1024, activation='relu', name='fc0')(X)
    X = Dropout(0.5)(X)

    X = Dense(512, activation='relu', name='fc1')(X)
    X = Dropout(0.5)(X)

    X = Dense(3, activation='softmax')(X)

    rec_model = keras.Model(inputs=X_input, outputs=X, name='RCNNmodel')

    summary = rec_model.summary()
    print(summary)

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=54000, decay_rate=0.9)
    opt = keras.optimizers.SGD(learning_rate=lr_schedule, name="SGD")
    rec_model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

    return rec_model

def trainRCNN(model, ep):

    X_train, y_train = preprocessing.loadTrainingSet()
    X_train, y_train = preprocessing.extract_patches(X_train, y_train, 3)

    X_train = keras.layers.ZeroPadding2D(padding=11, data_format='channels_last')(X_train)
    history = model.fit(X_train, y_train, epochs=ep, batch_size=1)
    utils.plot_Accuracy_Loss(history)
    model.save("RCNN.h5")