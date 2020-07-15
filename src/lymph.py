import IRRCNN
import CNN
import utils
import RCNN
import preprocessing
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    #X_train, X_test, y_train, y_test = preprocessing.create_dataset('/Users/brazzalenicola/Desktop/lymphoma')

    # X_train, X_test, y_train, y_test = preprocessing.load_dataset()
    '''
    plt.imshow(X_test[1,:,:,:])
    print(y_test[1])
    plt.show()
    '''

    #y_test_imgs = y_test

    '''
    plt.figure(0)
    plt.imshow(X_test[0,:,:,:])
    print(y_test[0])
    plt.show()
    plt.imshow(X_test[1,:,:,:])
    print(y_test[1])
    plt.show()
    

    X_train, y_train = preprocessing.extract_patches(X_train, y_train)
    X_test, y_test = preprocessing.extract_patches(X_test, y_test)

    labels, freqs = np.unique(y_train, return_counts=True)
    print("Labels in training dataset: ", labels)
    print("Frequencies in training dataset: ", freqs)

    # Normalize image vectors
    X_train = X_train / 255
    X_test = X_test / 255
    '''
    '''
    plt.figure(1)
    for i in range(324, 361):
        plt.imshow(X_test[i,:,:,:])
        print(y_test[i])
        plt.show()
    
    # Reshape
    y_train = y_train.T
    y_test = y_test.T

    # y_train = keras.utils.to_categorical(y_train)

    n_train = X_train.shape[0]
    print("number of training examples = " + str(n_train))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(y_test.shape))

    
    '''
    '''
    testmodel = keras.models.load_model('/Users/brazzalenicola/Desktop/CNN.h5')

    y_pred = []
    yPredictedProbs = testmodel.predict(X_test)
    yPredicted = yPredictedProbs.argmax(axis=1)
    print(yPredicted)
    yPredicted = np.split(yPredicted, 124)

    for pred in yPredicted:
        count_bins = np.bincount(pred)
        y_pred.append(np.argmax(count_bins))

    utils.print_confusion_matrix(y_test_imgs, y_pred, 3, ['FL', 'MCL', 'CLL'])
    print("Accuracy image wise: " + str((np.sum(y_test_imgs == y_pred) / 124) * 100) + " %")

    print("Evaluation patch-wise")
    preds = testmodel.evaluate(x=X_test, y=y_test)

    print("Test Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))
    '''
    input_size = 75
    RCNNmodel = RCNN.RCNN_model((input_size, input_size, 3))
    RCNN.RCNNtraining(RCNNmodel, 3)

