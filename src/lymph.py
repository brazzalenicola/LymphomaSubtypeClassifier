import CNN
import utils
import RCNN
import preprocessing
import sys

if __name__ == "__main__":

    if len(sys.argv) < 4:
        print('Mandatory parameters missing.')

    #CNN or RCNN
    nn_type = sys.argv[0]
    #RGB or GRAYSCALE or YUV or HSV
    color_space = sys.argv[1]
    #Training or evaluating
    method = sys.argv[2]
    #epochs
    ep = sys.argv[3]

    if nn_type == "CNN":

        cnn_model = CNN.CNN_model(input_shape=(64, 64, 3))

        if method == "train":
            CNN.trainCNN(cnn_model, ep, color_space)

        if method == "evaluate":
            CNN.evaluateCNN(cnn_model, color_space)

    elif nn_type == "RCNN":

        rec_model = RCNN.RCNN_model(input_shape=(64, 64, 3))

        if method == "train":
            CNN.trainCNN(rec_model, ep, color_space)

        if method == "evaluate":
            CNN.evaluateCNN(rec_model)





