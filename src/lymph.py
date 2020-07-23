import IRRCNN
import CNN
import utils
import RCNN
import preprocessing
import tensorflow as tf
from skimage.color import rgb2gray, rgb2hsv
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import pickle


if __name__ == '__main__':


    X_train, y_train = preprocessing.loadTrainingGraySet()
    X_train, y_train = preprocessing.extract_patches(X_train, y_train, 1)

    print(X_train.shape)
    gray_model = CNN.CNN_model((64, 64, 1))
    history = gray_model.fit(X_train, y_train, epochs=3, batch_size=32)
    utils.plot_Accuracy_Loss(history)
    gray_model.save("CNN_gray.h5")

    CNN.evaluateCNN(gray_model)

    #keras.models.load_model('/Users/brazzalenicola/Desktop/CNN.h5')
    #RCNNmodel = RCNN.RCNN_model((input_size, input_size, 3))
    #RCNN.RCNNtraining(RCNNmodel, 3)





