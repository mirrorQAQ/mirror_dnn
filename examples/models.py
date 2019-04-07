import layers

from activate import ReLU
from activate import SoftMax
from activate import Sigmoid
from pooling import MaxPooling


class DNN(layers.Module):
    def __init__(self, in_size, n_classes):
        self.h1 = layers.Linear(in_size, 128)
        self.h2 = layers.Linear(128, n_classes)

        self.relu = ReLU()
        self.softmax = SoftMax()
        self.sigmoid = Sigmoid()

    def forward(self, input):
        if len(input.shape) > 2:
            input = input.view(input.shape[0], -1)
        h1 = self.h1(input)
        # a = self.relu(h1)
        a = self.sigmoid(h1)
        h2 = self.h2(a)
        out = self.softmax(h2)
        return out

    def backward(self, dout, lr=0):
        dh2 = self.softmax.backward(dout)
        da = self.h2.backward(dh2, lr)
        # dh1 = self.relu.backward(da)
        dh1 = self.sigmoid.backward(da)
        self.h1.backward(dh1, lr)


class CNN(layers.Module):
    def __init__(self, in_channels, n_classes):
        self.in_channels = in_channels
        self.n_classes = n_classes

        self.conv1 = layers.Conv2D(in_channels, 3, 3)
        self.relu1 = ReLU()
        # self.sig1 = Sigmoid()
        self.pool1 = MaxPooling(kernel_size=2)

        self.fc1 = layers.Linear(3 * 13 * 13, 256)
        self.relu2 = ReLU()
        # self.sig2 = Sigmoid()

        self.fc2 = layers.Linear(256, n_classes)
        self.softmax = SoftMax()

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.fc1(x.view((x.shape[0], -1)))
        x = self.relu2(x)

        x = self.fc2(x)
        y = self.softmax(x)

        return y

    def backward(self, dy, lr):
        grad = self.softmax.backward(dy)
        grad = self.fc2.backward(grad, lr)
        grad = self.relu2.backward(grad)
        grad = self.fc1.backward(grad, lr)
        grad = self.pool1.backward(grad.view((dy.shape[0], 3, 13, 13)))
        grad = self.relu1.backward(grad)
        # print(grad.max())
        self.conv1.backward(grad, lr)
