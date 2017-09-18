import numpy as np
import random
import json
import sys


# Cross entropy cost function and its derivative
class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Defines cross entropy loss function.
        INPUT:
            a: activation (output) of last layer
            y: desired output i.e. true label
        RETURN:
            cross entropy between the 'real' distribution y
            and the 'unnatural' distribution a
        """
        return np.sum(np.nan_to_num(-y*np.log(a)))

    @staticmethod
    def delta(z, a, y):
        """Returns the error derivative i.e. dC/dz where
        C is the cross entropy between a and y and z is the
        softmax activation output
        INPUT:
            a: activation (output) of last layer
            y: desired output i.e. true label
            z: activation output (not used, included for reference)
        RETURN:
            error gradient from the output layer
            """
        return (a-y)

# Implement neural network
class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        """
        INPUT:
            sizes: list specifying the number of neurons.
             i.e sizes = [3,4,1] specifies three-layer network
                 with 3 neurons in the input layer, 4 in the
                 hidden layer and 1 neuron in the output layer
            cost: specifies the loss function
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost = cost
        self.initializer()

    def initializer(self):
        """Initialises weight matrix and bias vector using
        normal distribution.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """The feed forward step of hidden layers
        INPUT:
            a: input for the feed forward step
        OUTPUT:
            returns the activation of this layer when 'a' is the input
        """
        # all hidden layers use relu activation:
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            a = relu(np.dot(w, a) + b)
        # softmax output layer:
        b = self.biases[-1]
        w = self.weights[-1]
        a = softmax(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda, test_data,
            monitor_test_cost=False,
            monitor_test_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        """We train the network using stochastic gradient descent.
        INPUT:
            training_data: list of tuples (x,y) where x is the
            training sample input and y is the true label of x
            epochs: number of epochs to train over
            mini_batch_size: the size of each batch of training samples
            we use to train, that is we use the standard gradient descent
            on this batch of data
            eta: learning rate of gradient descent
            lmbda = L2 regularisation parameter
            test_data: test or validation set
            monitor_test_cost: set to 'True' if want test/validation cost every epoch
            monitor_test_accuracy: set to 'True' if want test/validation accuracy every epoch
            monitor_training_cost: set to 'True' if want train cost every epoch
            monitor_training_accuracy: set to 'True' if want train accuracy every epoch
        OUTPUT:
            returns a tuple containing four lists:
            1. per epoch costs on the evaluation data, accuracies of the
            evaluation data
            2. costs on the training data, accuracies on the training data
            also returns 'results_train' and 'results_test' which gives the predictions
            of the model - useful for confusion matrix
        """

        training_data = list(training_data)
        n = len(training_data)
        test_data = list(test_data)
        n_data = len(test_data)

        test_cost = []
        test_accuracy = []
        training_cost = []
        training_accuracy = []

        for j in range(epochs):
            random.shuffle(training_data) # randomise the ordering of training sample
            # Split whole training data into batches:
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))

            print("Epoch:", j)

            # Monitor cost and accuracy for reporting, note monitoring test performance slows down training:
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cross entropy loss (training set):", cost)
            if monitor_training_accuracy:
                results_train, accuracy = self.accuracy(training_data)
                training_accuracy.append(accuracy)
                print("Accuracy (training set):", accuracy/n)
            if monitor_test_cost:
                cost = self.total_cost(test_data, lmbda, convert=True)
                test_cost.append(cost)
                print("Cross entropy loss (test set):", cost)
            if monitor_test_accuracy:
                results_test, accuracy = self.accuracy(test_data)
                test_accuracy.append(accuracy)
                print("Accuracy (test set):", accuracy/n_data)

        return training_cost, training_accuracy, test_cost, \
               test_accuracy, results_train, results_test

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update weights and biases using SGD via backpropagation.
        INPUT:
            mini_batch: list of tuples (x,y) where x is the
            training sample input and y is the true label of x
            eta: learning rate
            lmbda: regularisation parameter
            n: size of whole training data
        OUTPUT:
            returns weights that have been updated using regularised
            gradient descent and biases update using just gradient descent
        """
        Db = [np.zeros(b.shape) for b in self.biases] # preallocate
        Dw = [np.zeros(w.shape) for w in self.weights] # preallocate
        for x, y in mini_batch: # for every tuple of input sample and corresponding true label
            delta_b, delta_w = self.backprop(x, y) # compute gradients
            # dnb is derivative of cost function wrt biases for given layer
            # delta_b contains dnb for each layer
            # dnw is derivative of cost function wrt weights for given layer
            # delta_w contains dnw for each layer
            # Db = sum up all 'dnb's across the samples in a given batch
            # Dw = sum up all 'dnw's across the samples in a given batch
            Db = [nb+dnb for nb, dnb in zip(Db, delta_b)]
            Dw = [nw+dnw for nw, dnw in zip(Dw, delta_w)]
        # Update the weights using regularised SGD (penalise large weights using L2):
        self.weights = [(w-w*eta*(lmbda/n)) - (eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, Dw)]
        # Update biases using SGD:
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, Db)]

    def backprop(self, x, y):
        """Implements backpropagation algorithm
        INPUT:
            x: input for current layer
            y: true label of x
        OUTPUT:
            returns a tuple (Db, Dw) which contains the gradients
            of the cost function wrt biases and weights for each layer.
            Db and Dw are layered lists of numpy arrays
        """
        # preallocate Db and Dw:
        Db = [np.zeros(b.shape) for b in self.biases]
        Dw = [np.zeros(w.shape) for w in self.weights]
        # Feed forward step:
        activation = x
        activations = [x] # store all activations by layer
        zs = [] # store all z=Wx+b vectors by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = relu(z) # ReLU layer
            activations.append(activation)

        delta = (self.cost).delta(zs[-1], activations[-1], y)
        Db[-1] = delta # since dz/db = 1 for all z and b
        Dw[-1] = np.dot(delta, activations[-2].transpose()) # since dz/dw = x
        # Going backwards from last layer: l=1 denotes last layer of neurons
        # l=2 means second last layer of neurons etc
        for l in range(2, self.num_layers):
            z = zs[-l]
            rp = relu_prime(z) # derivative of ReLU
            delta = np.dot(self.weights[-l+1].transpose(), delta) * rp
            Db[-l] = delta
            Dw[-l] = np.dot(delta, activations[-l-1].transpose())
        return (Db, Dw)

    def accuracy(self, data):
        """Evaluate the accuracy of the model on 'data'
        INPUT:
            data: training or validation or test set, including labels y
            convert: depends on whether 'data' is evaluation set or training
            if data is training, set to True and False otherwise
        OUTPUT:
            returns the accuracy on the specified data
        """
        results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]

        # Compute accuracy:
        result_accuracy = sum(1*(x==y) for (x, y) in results)
        return (results, result_accuracy)

    def total_cost(self, data, lmbda, convert=False):
        """Compute total cost of the model evaluated on 'data'
        INPUT:
            data: training or validation or test set, including labels y
            convert: depends on whether 'data' is evaluation set or training
            if data is training set to False and True otherwise
        OUTPUT:
            returns total cost on the specified data
        """
        cost = 0
        for (x, y) in data: # for every tuple
            a = self.feedforward(x)
            if convert:
                y = OneHotVector(y)
                cost += self.cost.fn(a, y) / len(data)
                cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights) # l2 loss
        return cost


    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

# Load a saved model
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

def OneHotVector(i):
    """
    Returns one hot vector with a 1 in the ith entry
    """
    v = np.zeros((10,1))
    v[i] = 1
    return v

def relu(z):
    """ReLU function"""
    return np.maximum(0, z)

def relu_prime(z):
    """Derivative of ReLU function"""
    return z*(z>0)

def softmax(z):
    """Softmax function"""
    #return (np.exp(z) / np.sum(np.exp(z)))
    # use log-sum-exp trick to prevent underflow/overflow issues
    m = np.max(z)
    return np.exp(z-m - np.log(np.sum(np.exp(z-m))))

