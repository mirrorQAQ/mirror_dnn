import torch

from utils import im2col
from layers import Module


class MaxPooling(Module):
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def forward(self, feature_map):
        batch_size = feature_map.shape[0]
        in_channels = feature_map.shape[1]
        in_height = feature_map.shape[2]
        in_width = feature_map.shape[3]

        self.in_height = in_height
        self.in_width = in_width

        self.in_channels = in_channels

        out_width = in_width // self.kernel_size
        out_height = in_height // self.kernel_size

        self.out_width = out_width
        self.out_height = out_height

        self.cut_in_h = self.out_height * self.kernel_size
        self.cut_in_w = self.out_width * self.kernel_size

        out_channels = in_channels

        feat_mat = im2col(feature_map[:, :, :self.cut_in_h, :self.cut_in_w], self.kernel_size, out_height, out_width,
                          in_channels, stride=self.kernel_size)  # R(ic*k*k,bt*oh*ow)
        feat_mat = feat_mat.view(out_channels, self.kernel_size ** 2, -1)

        out_map, self.max_ind = feat_mat.max(dim=1)
        #
        # print(feat_mat)
        # print("feat_mat\n")
        #
        # print(self.max_ind)
        # print("max_ind\n")

        out_map = out_map.view(out_channels, batch_size, out_height, out_width).transpose(1, 0).contiguous()

        return out_map

    def backward(self, dl_out):
        batch_size, out_channels, oh, ow = dl_out.shape

        dout_in = torch.zeros((self.in_channels, self.kernel_size ** 2, self.out_height * self.out_width * batch_size),
                              dtype=dl_out.dtype)
        # print(self.max_ind)

        inds0 = torch.arange(self.in_channels)
        inds2 = torch.arange(dout_in.shape[2])

        n0 = inds0.numel()
        n1 = self.max_ind.numel()
        n2 = inds2.numel()

        inds1 = self.max_ind.view(-1)
        inds0 = inds0.view(-1, 1).expand(n0, n2).contiguous().view(-1)
        inds2 = inds2.repeat(n0)

        # print(inds0)
        # print(inds1)
        # print(inds2)
        dout_in[inds0, inds1, inds2] = 1
        # print(dl_in)
        # print("dln\n")

        dout_in = dout_in.view((self.in_channels, self.kernel_size ** 2, batch_size, self.out_height, self.out_width))

        dout_in = dout_in.permute((2, 0, 3, 4, 1)).contiguous()
        dout_in = dout_in.view((batch_size, self.in_channels, self.out_height, self.out_width,
                                self.kernel_size, self.kernel_size)).permute((0, 1, 2, 4, 3, 5))

        dout_in = dout_in.contiguous().view((batch_size, self.in_channels, self.cut_in_h, self.cut_in_w))

        dl_out = dl_out.view(batch_size, out_channels, oh * ow, 1).repeat(1, 1, 1, self.kernel_size).contiguous()
        dl_out = dl_out.view(batch_size, out_channels, oh, -1).repeat(1, 1, 1, self.kernel_size).contiguous()
        dl_out = dl_out.view(batch_size, out_channels, self.cut_in_h, self.cut_in_w)

        dl_in = dl_out * dout_in

        # print(dl_in)
        if self.in_height != self.cut_in_h or self.in_width != self.cut_in_w:
            padding_din = torch.zeros((batch_size, self.in_channels, self.in_height, self.in_width))
            padding_din[:, :, :self.cut_in_h, :self.cut_in_w] = dl_in
            return padding_din

        return dl_in


if __name__ == '__main__':
    bt, c, h, w = 2, 3, 4, 4
    x = torch.randn(bt * c * h * w).reshape((bt, c, h, w))
    maxpooling = MaxPooling(2)

    print(x)
    print("x\n")
    out = maxpooling(x)

    print(out)
    print("out\n")

    dout = torch.arange(bt * c * h // 2 * w // 2).view((bt, c, h // 2, w // 2))

    print(maxpooling.backward(dout))
    print("grad\n")
