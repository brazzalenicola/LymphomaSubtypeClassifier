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


    #X_train, y_train = preprocessing.loadTrainingGraySet()
    #X_train, y_train = preprocessing.extract_patches(X_train, y_train, 1)

    #print(X_train.shape)
    #gray_model = CNN.CNN_model((64, 64, 1))
    #history = gray_model.fit(X_train, y_train, epochs=3, batch_size=32)
    #utils.plot_Accuracy_Loss(history)
    #gray_model.save("CNN_gray.h5")

    #CNN.evaluateCNN(gray_model)

    #input_size = 64
    #keras.models.load_model('/Users/brazzalenicola/Desktop/CNN.h5')
    #RCNNmodel = RCNN.RCNN_model((input_size, input_size, 3))
    #RCNN.RCNNtraining(RCNNmodel, 3)

    '''
    X_train, y_train = preprocessing.loadTrainingHSVSet()
    X_train, y_train = preprocessing.extract_patches(X_train, y_train, 3)
    hsv_model = CNN.CNN_model((64, 64, 3))
    history = hsv_model.fit(X_train, y_train, epochs=3, batch_size=32)
    utils.plot_Accuracy_Loss(history)
    hsv_model.save("CNN_hsv.h5")
    '''


    rcnn = RCNN.RCNN_model((64,64,3))
    history = RCNN.trainRCNN(rcnn, 70)

    '''
    print("Evaluation patch-wise:")
    preds = rcnn.evaluate(x=X_test, y=y_test)

    print("Test Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))

    y_pred = []
    yPredictedProbs = hsv_model.predict(X_test)
    yPredicted = yPredictedProbs.argmax(axis=1)
    yPredicted = np.reshape(yPredicted, (yPredicted.shape / 336, 336))

    for pred in yPredicted:
        count_bins = np.bincount(pred)
        y_pred.append(np.argmax(count_bins))

    print("Accuracy image wise: " + str((np.sum(y_test_lab == y_pred) / 124) * 100) + " %")
    '''







