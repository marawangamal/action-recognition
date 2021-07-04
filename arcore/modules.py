import torch.nn as nn
import torch
import torch.nn.functional as F


# BASIC MODULES

class LayerModules:
    """ Building blocks for FreqModel."""

    def cbrp(self, args):
        return CBRP(*args)

    def esa(self, args):
        return ESA(*args)

    def lbrd(self, args):
        return LBRD(*args)

    def gap(self, args):
        return GAP(*args)

    def tp(self, args):
        return TP(*args)

    def reshape(self, args):
        return RESHAPE(*args)

    def collapse(self, args):
        return COLLAPSE(*args)

    def tcp(self, args):
        return TCP(*args)


class CBRP(nn.Module):
    """ conv-batchnorm-relu-maxpool on (4/5)-Dimensional input"""

    def __init__(self, in_f, out_f, k_size, stride, padd, bn, relu, pool, log_output=False):
        super(CBRP, self).__init__()

        self.log_output = log_output

        if (in_f > 0 or bnr or pool):
            self.cbrp = nn.Sequential()

        if (in_f > 0):
            self.cbrp.add_module("conv",
                                 nn.Conv2d(in_channels=in_f, out_channels=out_f, stride=stride, kernel_size=k_size,
                                           padding=padd))
        if (bn):
            self.cbrp.add_module("bn", nn.BatchNorm2d(out_f))
        if (relu):
            self.cbrp.add_module("relu", nn.LeakyReLU())
        if (pool):
            self.cbrp.add_module("pool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False))

    def forward(self, x):
        """ performs cbrp on (4/5)-Dimensional input
        args:
            x: (tensor) [N, C, H, W] or [N, T, C, H, W]
        returns:
            x: (tensor) [N, C', H', W'] or [N, T, C', H', W']
        """

        if (len(x.shape) > 4):
            N, T, C, H, W = x.shape
            out = self.cbrp(x.view(N * T, C, H, W))
            NT, C, H, W = out.shape
            out = out.view(N, T, C, H, W)

        else:
            out = self.cbrp(x)
        return out


class LBRD(nn.Module):
    """linear-batchnorm-relu-dropout"""

    def __init__(self, in_f, out_f, bn, relu, dropout, log_output=False):
        super(LBRD, self).__init__()
        self.log_output = log_output

        self.lbrd = nn.Sequential()

        if (in_f > 0):
            self.lbrd.add_module("linear", nn.Linear(in_f, out_f))
        if (bn):
            self.lbrd.add_module("bn", nn.BatchNorm1d(out_f))
        if (relu):
            self.lbrd.add_module("relu", nn.ReLU())
        if (dropout):
            self.lbrd.add_module("dropout", nn.Dropout(dropout))

    def forward(self, x):
        """
        args:
            x: (tensor) [N, T, in_f]
        returns:
            x: (tensor) [N, T, out_f]
        """
        if (len(x.shape) > 2):
            N, T, in_f = x.shape
            out = self.lbrd(x.view(N * T, -1)).view(N, T, -1)
        else:
            out = self.lbrd(x)

        return out


# SPEICAL

class ESA(nn.Module):
    """ equivariant self-attention"""

    def __init__(self, c_in, skip, attn_cfg, log_output=False):
        super(ESA, self).__init__()
        self.log_output = log_output

        self.skip = skip
        self.conv_attention = ConvAttention(c_in, attn_cfg)

    def forward(self, x):
        """ returns same shape as input, after performing ESA
        args:
            x: (tensor) [N, T, C, H, W]
        returns
            x: (tensor) [N, T, C, H, W]
        """

        N, T, C, H, W = x.shape

        ca_out = self.conv_attention(x, x, x)[0]

        if (self.skip):
            if (ca_out.shape != x.shape):
                raise Exception("Wrong size for skip connection")
            ca_out = ca_out + x

        return ca_out


class TP(nn.Module):
    """ temporal pooling """

    def __init__(self, type='avg', log_output=False):
        super(TP, self).__init__()
        self.log_output = log_output
        self.type = type

    def forward(self, x):
        """
        args:
            x: (tensor) [N, T, *]
        returns:
            out: (tensor) [N, *]
        """

        N, T = x.size(0), x.size(1)

        if (self.type in "avg"):
            out = torch.mean(x, dim=1)
        elif (self.type in "center"):
            out = x[:, int(T / 2), ...]
        elif ("color" in self.type):
            out = self.color_pool(x)

        return out

    def color_pool(self, x):
        """
        args:
            x: (tensor) [N, T, C, H, W]
        returns:
            x: (tensor) [N, C, H, W]
        """

        #
        pass


class TCP(nn.Module):
    """ temporal color pooling """

    def __init__(self, dim_T, log_output=False):
        super(TCP, self).__init__()
        self.log_output = log_output
        self.dim_T = dim_T  # number of frames
        self.Cmap = torch.tensor([[t / self.dim_T, 1 - t / self.dim_T] for t in range(self.dim_T)]).T  # [2xT]
        self.Cmap = self.Cmap.view(1, 2, self.dim_T, 1, 1, 1)

    def forward(self, x):
        """
        args:
            x: (tensor) [N, T, C, H, W]
        returns:
            y: (tensor) [N, 2C, H, W]
        """

        N, T, C, H, W = x.shape
        device = x.device
        y = x.unsqueeze(1) * self.Cmap.to(device)  # [N,1,T, C, H, W] * [1,2,T, 1, 1, 1]
        y = torch.sum(y, dim=2).reshape(N, 2 * C, H, W)

        return y


# RESHAPING

class RESHAPE(nn.Module):
    """ reshape from [NT, *] to [N, T, *]. Useful after batching """

    def __init__(self, num_segs=3, log_output=False):
        super(RESHAPE, self).__init__()
        self.log_output = log_output
        self.num_segs = num_segs

    def forward(self, x):
        """
        args:
            x: (tensor) [NT, *]
        returns:
            out: (tensor) [N, T, *]
        """

        dims = x.shape
        new_dims = [-1, self.num_segs, *dims[1:]]
        x.view(*new_dims)

        return x


class COLLAPSE(nn.Module):
    """ reshape last `num_dims` dimensions [N, T, *d] to [N, T, -1], useful for transition to fc layer """

    def __init__(self, num_dims, side='back', log_output=False):
        super(COLLAPSE, self).__init__()
        self.log_output = log_output
        self.num_dims = num_dims
        self.side = side

    def forward(self, x):
        """
        args:
            x: (tensor) [N, T, C, H, W]
        returns:
            out: (tensor) [N, T, D] (for num_dims=3, and side='back')
        """

        if (self.side == 'back'):
            outdims = [x.shape[d] for d in range(len(x.shape) - self.num_dims)]
            outdims.append(-1)
            x = x.view(*outdims)

        else:
            outdims = [-1]
            outdims = [x.shape[d] for d in range(self.num_dims, len(x.shape))]
            x = x.view(*outdims)
        return x


# POOLING

class GAP(nn.Module):
    """ spatial pooling on (4/5)-dimensional input """

    def __init__(self, x=None, log_output=False):
        super(GAP, self).__init__()
        self.log_output = log_output

    def forward(self, x):
        """
        args:
            x: (tensor) [N, T, C, H, W] or [N, C, H, W]
        returns:
            x: (tensor) [N, T, C] or [N, C]
        """
        if (len(x.shape) > 4):
            N, T, C, H, W = x.shape
            x = F.avg_pool2d(x.view(N * T, C, H, W), kernel_size=(H, W)).view(N, T, C)
        else:
            N, C, H, W = x.shape
            out = F.avg_pool2d(x, kernel_size=(H, W)).view(N, C)

        return out
