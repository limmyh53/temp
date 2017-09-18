import numpy as np
import Net
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import confusion_matrix


# Import data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
# Organise the imported data into tuples of images and the corresponding label:
# Training set
X_train = [np.reshape(x, (784, 1)) for x in mnist.train.images]
Y_train = [np.reshape(y, (10, 1)) for y in mnist.train.labels]
train_data = zip(X_train, Y_train)
# Validation set
X_validation = [np.reshape(x, (784, 1)) for x in mnist.validation.images]
Y_validation = [np.reshape(y, (10, 1)) for y in mnist.validation.labels]
validation_data = zip(X_validation, Y_validation)
# Test set
X_test = [np.reshape(x, (784, 1)) for x in mnist.test.images]
Y_test = [np.reshape(y, (10, 1)) for y in mnist.test.labels]
test_data = zip(X_test, Y_test)


# Implement the neural network
net = Net.Network([784, 256, 256, 10], cost=Net.CrossEntropyCost)
results = net.SGD(train_data, epochs=30, mini_batch_size=100, eta=0.01, lmbda=0.001,
        test_data=test_data,
        monitor_test_cost=False,
        monitor_test_accuracy=True,
        monitor_training_cost=False,
        monitor_training_accuracy=True)[5]


results = np.array(results)
confusion = confusion_matrix(y_true=results[:,1],
                       y_pred=results[:,0], labels=np.arange(10))

print("Confusion matrix: \n", confusion)

net.save(filename='.\saved_models\P2c')
