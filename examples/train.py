import numpy as np
import matplotlib.pyplot as plt
import pickle
from examples import models
import loss
import torch

batch_size = 100
epochs = 300
learning_rate = 0.08

in_size = 28 * 28
n_classes = 10

mnist_images = np.load("data/mnist_images.npy")
test_images = mnist_images[3000:]  # 500 images
train_images = mnist_images[:3000]  # 4500 images
mnist_labels = np.load("data/mnist_labels.npy")
test_labels = mnist_labels[3000:]
train_labels = mnist_labels[:3000]


def plot_mnist(remove_border=False):
    """ Plot the first 40 data points in the MNIST train_images. """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    for i in range(4 * 10):
        plt.subplot(4, 10, i + 1)
        if remove_border:
            plt.imshow(train_images[i, 4:24, 4:24])
        else:
            plt.imshow(train_images[i])


train_X = train_images.reshape([3000, 1, 28, 28])
train_Y = np.zeros([3000, 10])
row_inds = tuple(range(3000))
col_inds = train_labels
train_Y[row_inds, col_inds] = 1

# LR = LogisticRegression(784, 10, learning_rate)
criterion = loss.MSE()


# train_X = train_X[:500]
# train_Y = train_Y[:500]


def train(cnn, X, Y):
    n_samples = Y.shape[0]
    losses = []

    for e in range(epochs):
        inds = list(range(n_samples))
        np.random.shuffle(inds)
        i = 0
        total_loss = 0

        # print(cnn.conv1.kernel)

        while i < n_samples:
            batch_inds = inds[i: i + batch_size]
            batch_x = torch.from_numpy(X[batch_inds]).float()
            batch_y = torch.from_numpy(Y[batch_inds]).float()

            pred_y = cnn(batch_x)
            # print(pred_y)
            loss = torch.sum(criterion(pred_y, batch_y)).item()
            # print("batch loss: ", loss)
            total_loss += loss

            dy = criterion.backward()
            cnn.backward(dy, learning_rate)
            # print(cnn.conv1.kernel)

            i += batch_size

        mean_loss = total_loss / n_samples
        if len(losses):
            if abs(mean_loss - losses[-1]) < 0.00000:
                break
        pred_y = cnn(torch.from_numpy(X).float())
        _, pred_y = torch.max(pred_y, dim=1)
        _, true_y = torch.max(torch.from_numpy(Y), dim=1)
        pred_y = pred_y.long()
        true_y = true_y.long()

        res = pred_y == true_y
        acc = torch.sum(res).item() / Y.shape[0]

        print("epoch: {}, loss: {}, acc: {}".format(e, mean_loss, acc))
        losses.append(mean_loss)

    print("save model to {}".format("./cnn.pkl"))
    with open("./cnn.pkl", 'wb') as f:
        pickle.dump(cnn, f)

    return losses


if __name__ == '__main__':
    # LR = LogisticRegression(in_size, n_classes, 0.01)
    # losses = train(LR, train_X, train_Y)

    m = models.CNN(1, n_classes)
    losses = train(m, train_X, train_Y)
    plt.plot(range(len(losses)), losses)
    plt.show()
