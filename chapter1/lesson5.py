import sys
from chapter1 import c1_utils
from mxnet import ndarray as nd
from mxnet import gluon
from mxnet import autograd as autograd

batch_size = 256
train_data, test_data = c1_utils.load_data_fashion_mnist(batch_size)

num_inputs = 28 * 28
num_outputs = 10

num_hidden = 784
weight_scale = .05

W1 = nd.random_normal(shape=(num_inputs, num_hidden), scale=weight_scale)
b1 = nd.zeros(num_hidden)

W2 = nd.random_normal(shape=(num_hidden, num_outputs), scale=weight_scale)
b2 = nd.zeros(num_outputs)

params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()


def relu(X):
    return nd.maximum(X, 0)


def net(X):
    X = X.reshape((-1, num_inputs))
    h1 = relu(nd.dot(X, W1) + b1)
    output = nd.dot(h1, W2) + b2
    return output


softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

learning_rate = .5
for epoch in range(10):
    train_loss = 0.
    train_acc = 0
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        c1_utils.SGD(params, learning_rate / batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += c1_utils.accuracy(output, label)
    test_acc = c1_utils.evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss / len(train_data),
        train_acc / len(train_data), test_acc))
