# import numpy as np
import torch

from utils import im2col
from utils import kernel2row


class Module(object):
    def forward(self, *inputs, **kwargs):
        raise NotImplementedError()

    def backward(self, *inputs, **kwargs):
        raise NotImplementedError()

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)


class Linear(Module):
    # in_size : X`s feature number
    # out_size : Z`s features number
    # batch_size :  how many sample in one batch

    def __init__(self, in_size, out_size):
        self.w = torch.randn(in_size, out_size).float()  # with R(in_size,out_size)
        self.b = torch.zeros(out_size, 1).float() + 0.1 # with R(out_size,1)
        self.cur_x = None
        self.cur_z = None

    def forward(self, batch_x):
        """
        Args:
            batch_x: with R(batch_size,in_size)
        Returns:
            z : with R(batch_size,out_size)
        """

        self.cur_x = batch_x
        z = torch.matmul(batch_x, self.w) + self.b.transpose(1, 0)

        self.cur_z = z

        return z

    def backward(self, dl_z, lr):
        """
        Args:
            z: R(batch_size, out_size)
                行: 每一个样本的输入神经元向量
                列: 同一个输入神经元在不同的样本中
            dl_z: R(batch_size, out_size)
        Returns:
            dl_x :R(batch_size,in_size)

        """
        batch_size = self.cur_x.shape[0]
        in_size = self.cur_x.shape[1]
        out_size = self.cur_z.shape[1]

        dz_w = torch.unsqueeze(self.cur_x, 1)
        dz_b = torch.ones((batch_size, out_size, 1))

        dz_x = torch.unsqueeze(self.w, 0)  # R(1(broadcast to bt),out,in)
        dl_x = torch.empty((batch_size, in_size))
        for i in range(batch_size):
            dl_x[i] = torch.matmul(dl_z[i], dz_x[0].transpose(1, 0))

        dl_z = torch.unsqueeze(dl_z, 2)  # R(bt,out,1)

        dl_w = dl_z * dz_w
        dl_b = dl_z * dz_b

        # update
        self.step(dl_w, dl_b, lr)

        return dl_x

    def step(self, dw, db, lr):
        self.w -= torch.mean(dw, dim=0).transpose(1, 0) * lr
        self.b -= torch.mean(db, dim=0) * lr


class Conv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.kernel = self._init_weights(self.kernel_size, self.out_channels,
                                         self.in_channels)  # R（out_channels,in_channels,k,k）

        self.in_c = None
        self.in_h = None
        self.in_w = None

    def _init_weights(self, kernel_size, out_channels, in_channels):
        return torch.randn(out_channels, in_channels, kernel_size, kernel_size).float()

    def forward(self, feature_map):
        # feature_map : R(batch_size,in_channels,in_width,in_height)
        batch_size, in_channels, height, width = feature_map.shape

        self.in_c = in_channels
        self.in_h = height
        self.in_w = width

        if self.padding > 0:
            padding_h = height + 2 * self.padding
            padding_w = width + 2 * self.padding
            padding_feat = torch.zeros((batch_size, in_channels, padding_h, padding_w), dtype=feature_map.dtype)
            padding_feat[:, :, self.padding:self.padding + height, self.padding:self.padding + width] = feature_map
            feature_map = padding_feat

        h_steps = (height - self.kernel_size + 2 * self.padding) // self.stride + 1
        v_steps = (width - self.kernel_size + 2 * self.padding) // self.stride + 1

        self.feat_mat = im2col(feature_map, self.kernel_size, h_steps, v_steps, self.in_channels, self.stride)
        self.kernel_mat = kernel2row(self.kernel)

        outmap = torch.matmul(self.kernel_mat, self.feat_mat)
        outmap = outmap.view((self.out_channels, batch_size, h_steps, v_steps)).permute((1, 0, 2, 3))

        return outmap

    def backward(self, dl_out, lr):
        batch_size, out_c, out_h, out_w = dl_out.shape
        padding_h = (self.in_h - 1) * self.stride + self.kernel_size - 2 * self.padding - out_h
        padding_w = (self.in_w - 1) * self.stride + self.kernel_size - 2 * self.padding - out_w

        padding_h //= 2
        padding_w //= 2

        padding_dout = torch.zeros((batch_size, out_c, out_h + 2 * padding_h, out_w + 2 * padding_w),
                                   dtype=dl_out.dtype)
        padding_dout[:, :, padding_h: padding_h + out_h, padding_w:padding_w + out_w] = dl_out
        rotated_kernels = self.kernel.transpose(1, 0).flip(2).flip(3)  # (oc, ic, k, k) -> (ic, oc, k, k)

        padding_dout2col = im2col(padding_dout, self.kernel_size, self.in_w, self.in_h, self.out_channels, self.stride)
        rotated_kernels2row = kernel2row(rotated_kernels)

        dl_in = torch.matmul(rotated_kernels2row, padding_dout2col)  # (ic, batch_size*in_h*in_w)
        dl_in = dl_in.reshape((self.in_channels, batch_size, self.in_h, self.in_w)).transpose(1, 0)

        dl_out = dl_out.permute(1, 0, 2, 3).contiguous().view(self.out_channels, -1)
        feat_mat = self.feat_mat.transpose(1, 0)

        dl_k = torch.matmul(dl_out, feat_mat) / batch_size
        dl_k = dl_k.view((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))

        self.kernel -= lr * dl_k

        return dl_in

