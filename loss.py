# import numpy as np
import torch
import layers


class CrossEntropy(layers.Module):
    def __init__(self):
        self.pre_y = None

    def forward(self, pre_y, true_y):
        """
        Args:
            pre_y: R(bt,in_size)
            true_y: R(bt,in_size)

        Returns:
            loss: R(bt,1)
        """
        batch_size = pre_y.shape[0]

        log_y = torch.log(pre_y)
        loss = -1 * (true_y * log_y)

        self.pre_y = pre_y

        loss = torch.sum(loss, dim=1)  # R(bt,)
        loss = torch.reshape(loss, (batch_size, 1))  # R(bt,1)

        return loss

    def backward(self, ):
        dl_a = -1 * torch.log(self.pre_y)  # R(batch_size,in_size)
        return dl_a


class MSE(layers.Module):

    def __init__(self):
        self.pred_y = None
        self.true_y = None

    def forward(self, pred_y, true_y):
        """
        Args:
            pred_z: R(batch_size, in_size)
            true_y: R(batch_size, in_size)
        Returns:
            loss: R(batch_size, 1)
        """

        self.true_y = true_y
        self.pred_y = pred_y

        dif = self.pred_y - self.true_y
        loss = torch.sum(dif ** 2, dim=1)

        return loss.reshape((self.pred_y.shape[0], 1))

    def backward(self):
        dl_y = self.pred_y - self.true_y  # dl_y : R(batch_size, out_size)
        return dl_y
