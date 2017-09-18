import numpy as np
import Net
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import confusion_matrix
from Net import softmax, relu

# Import data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

X_test = [np.reshape(x, (784, 1)) for x in mnist.test.images]
Y_test = [np.reshape(y, (10, 1)) for y in mnist.test.labels]
test_data = zip(X_test, Y_test)

# Load saved data
P2b = Net.load(filename='.\saved_models\P2b')
w = P2b.weights
b = P2b.biases


# feedforward step to implement trained network
def feedforward(a, weights, biases):
    """The feed forward step of hidden layers
    INPUT:
        a: input for the feed forward step
    OUTPUT:
        returns the activation of this layer when 'a' is the input
    """
    # all hidden layers use relu activation:
    for w, b in zip(weights[:-1], biases[:-1]):
        a = relu(np.dot(w, a) + b)
    # softmax output layer:
    b = biases[-1]
    w = weights[-1]
    a = softmax(np.dot(w, a) + b)
    return a


# Get predictions
ps = []
for xx in X_test:
    p = feedforward(xx, w, b)
    ps.append(p)

y_pred = np.argmax(ps, axis=1)
y_true = np.argmax(Y_test, axis=1)
print("Test accuracy:", np.sum(y_true==y_pred)/10000)

confusion = confusion_matrix(y_true=y_true,
                        y_pred=y_pred, labels=np.arange(10))

print("Confusion matrix: \n", confusion)


# PREDICTION:
# If want to make prediction replace the variable 'x_new' below with some input
x_new = np.random.randn(784,1)
y_new = feedforward(x_new, w, b)
print("Prediction probability on x_new: \n", y_new)
