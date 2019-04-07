# import numpy as np
import torch
import layers


class SoftMax(layers.Module):

    def __init__(self):
        self.cur_z = None
        self.exp_z = None
        self.preY = None

    def forward(self, z):
        """
        Args:
            z: with R(batch_size, in_size)
               in_size = n_classes
        Returns:
            a: with R(batch_size, out_size)
                out_size =  in_size
        """
        z = z - torch.unsqueeze(z.min(dim=1)[0], dim=1)
        self.cur_z = z
        self.exp_z = torch.exp(-z)
        dom = torch.sum(self.exp_z, dim=1).reshape((z.shape[0], 1))

        a = self.exp_z / dom
        self.preY = a
        return a

    def backward(self, dl_y):
        """
        Args:po
            dl_y:  R(batch_size,n_classes) n_classes =  in_size
        Returns:
            dl_z:
        """
        batch_size = self.cur_z.shape[0]
        in_size = self.cur_z.shape[1]
        dy_z = torch.empty((batch_size, in_size, in_size))

        for i in range(batch_size):
            diag = torch.diag(self.preY[i])
            sing_y_sample = torch.unsqueeze(self.preY[i], dim=0)  # R(1，in_size)
            sample_grad = torch.matmul(sing_y_sample.transpose(1, 0), sing_y_sample)  # R(in_size,in_size)

            sample_grad = sample_grad - diag

            dy_z[i] = sample_grad
        dy_z = dy_z

        dl_z = torch.empty((batch_size, in_size))

        for i in range(batch_size):
            # dl_y[i] : R(1,in_size)
            # dy_z[i] : R(in_size,in_size)

            dl_z[i] = torch.matmul(dl_y[i], dy_z[i])  # R(1,in_size)

        return dl_z  # R(batch_size, in_size)


class Sigmoid(layers.Module):
    def __init__(self):
        self.a = None

    def forward(self, z):
        """
        Args:
            z: R(batch_size, in_size)

        Returns:
            a: R(batch_size, in_size)
        """
        self.a = 1 / ((1 + torch.exp(-z)) + 0.01)
        return self.a

    def backward(self, dl_a):
        """
        dl_a:  R(batch_size,out_size)
        Returns:
            dl_z: R(batch_size, in_size)
        """

        da_z = self.a * (1 - self.a) + 0.01  # R(batch_size,in_size)
        dl_z = dl_a * da_z

        return dl_z  # R(batch_size,in_size)


class ReLU(layers.Module):
    def __init__(self):
        self.a = None

    def forward(self, z):
        # z: R(batch_size,in_size)
        a = z.clone()
        a[a < 0] = 0
        self.a = a
        return a

    def backward(self, dl_a):
        """
        Args:
            dl_a:  R(batch_size,out_size)
        Returns:
            dl_z: R(batch_size,in_size)
        """

        # batch_size = self.a.shape[0]
        # in_size = self.a.shape[1]

        da_z = self.a
        da_z[da_z > 0] = 1

        dl_z = dl_a * da_z

        return dl_z
