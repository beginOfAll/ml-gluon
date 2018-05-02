import sys
from chapter1 import c1_utils
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd

batch_size = 256
train_data, test_data = c1_utils.load_data_fashion_mnist(batch_size)

net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(10))
net.initialize()
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

for epoch in range(20):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        temp_train_acc = c1_utils.accuracy(output, label)
        # print(temp_train_acc)
        train_acc += temp_train_acc

    test_acc = c1_utils.evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))