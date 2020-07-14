import IRRCNN
import CNN
import utils
import preprocessing

if __name__ == '__main__':

    #create_dataset('/Users/brazzalenicola/Desktop/lymphoma')

    X_train, X_test, y_train, y_test = preprocessing.load_dataset()

    y_test_imgs = y_test

    X_train, y_train = preprocessing.extract_patches(X_train, y_train)
    X_test, y_test = preprocessing.extract_patches(X_test, y_test)

    # Normalize image vectors
    X_train = X_train / 255
    X_test = X_test / 255

    # Reshape
    y_train = y_train.T
    y_test = y_test.T

    #y_train = keras.utils.to_categorical(y_train)

    n_train = X_train.shape[0]
    print("number of training examples = " + str(n_train))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(y_test.shape))

    input_size = 64
    #testmodel = keras.models.load_model('/Users/brazzalenicola/Desktop/IRRCNN.h5')

    #opt = keras.optimizers.Adagrad(learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07, name="Adagrad")
    #opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name="Adam")
    #cnnmodel = CNN.CNN_model((input_size, input_size, 3))





