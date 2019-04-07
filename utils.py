import torch


def im2col(imgs, kennel_size, h_steps, v_steps, in_channels, stride):
    batch_size = imgs.shape[0]

    mat = torch.empty((in_channels * kennel_size * kennel_size, h_steps * v_steps * batch_size), dtype=imgs.dtype)
    for bt in range(batch_size):
        for h in range(h_steps):
            for v in range(v_steps):
                mat[:, bt * h_steps * v_steps + h * v_steps + v] = make_col(imgs, bt, kennel_size, h, v, stride)

    return mat


def kernel2row(kernels):
    out_channels = kernels.shape[0]
    return kernels.view(out_channels, -1)


def make_col(imgs, bt, kennel_size, h, v, stride):
    return imgs[bt, :, h * stride:h * stride + kennel_size, v * stride:v * stride + kennel_size].contiguous().view(-1)
