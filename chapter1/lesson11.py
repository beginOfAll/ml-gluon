from mxnet.gluon import nn

net = nn.Sequential()
drop_prob1 = 0.8
drop_prob2 = 0.1

with net.name_scope():
    net.add(nn.Flatten())
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dropout(drop_prob1))
    net.add(nn.Dense(256, activation="relu"))
    net.add(nn.Dropout(drop_prob2))
    net.add(nn.Dense(10))
net.initialize()

from mxnet import nd
from mxnet import autograd
from mxnet import gluon
from chapter1 import c1_utils

batch_size = 256
train_data, test_data = c1_utils.load_data_fashion_mnist(batch_size)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),
                        'sgd', {'learning_rate': 0.5})

for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += c1_utils.accuracy(output, label)

    test_acc = c1_utils.evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data),
        train_acc/len(train_data), test_acc))