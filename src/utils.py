from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_Accuracy_Loss(history):
    plt.figure()
    plt.plot(history.history['loss'], label='Train loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Plot accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.show()

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
