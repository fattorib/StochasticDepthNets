import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch.nn.init as init
from torch import Tensor
from scipy.stats import bernoulli


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


# ----------Building Blocks----------


class ResidualBlock(nn.Module):
    # One full block of a given filter size
    def __init__(self, in_filters, out_filters, N, layer_probs=[], downsample=True):
        super(ResidualBlock, self).__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.N = N

        # TBD whether this is useful
        self.np_lp = layer_probs.clone().detach()

        self.downsample = downsample
        self.conv_block = nn.Sequential(nn.Conv2d(
            self.in_filters, self.in_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_filters),
            nn.ReLU(),
            nn.Conv2d(
            self.in_filters, self.in_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_filters)
        )

        self.residual_block = nn.ModuleList([copy.deepcopy(
            self.conv_block) for _ in range(self.N-1)])
        self.bn = nn.BatchNorm2d(out_filters, affine=True)

        # Downsample using stride of (2,2)
        if self.downsample:
            self.final_block = nn.Sequential(nn.Conv2d(
                self.in_filters, self.in_filters, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.in_filters),
                nn.ReLU(),
                nn.Conv2d(self.in_filters, self.out_filters,
                          kernel_size=3, stride=(2, 2), padding=1, bias=False),
                nn.BatchNorm2d(self.out_filters)
            )
        else:
            self.final_block = nn.Sequential(nn.Conv2d(
                self.in_filters, self.in_filters, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.in_filters),
                nn.ReLU(),
                nn.Conv2d(
                self.in_filters, self.in_filters, kernel_size=3, stride=(1, 1), padding=1, bias=False),
                nn.BatchNorm2d(self.in_filters))
        self.apply(_weights_init)

        # Register probs as buffers to avoid costly data transfers between cpu and gpu
        self._register(layer_probs)

    def _register(self, layer_probs):
        self.register_buffer('layer_probs', layer_probs)

    def forward(self, x: Tensor) -> Tensor:

        if self.training:
            b_l = torch.bernoulli(self.layer_probs)

            for i in range(len(self.residual_block)):
                if torch.rand(1)[0] <= self.np_lp[i]:
                    residual = x
                    x = self.residual_block[i](x)
                    x += residual
                    x = F.relu(x)
                else:
                    x = F.relu(x)
            # Perform downsampling on last layer and add final residual

            residual = x
            x = b_l[-1]*self.final_block(x)
            if self.downsample:
                x += self.pad_identity(residual)
            else:
                # Don't downsample on final layer
                x += residual
            return F.relu(x)

        else:
            b_l = self.layer_probs
            for i in range(len(self.residual_block)):
                residual = x
                x = b_l[i]*self.residual_block[i](x)
                x += residual

                x = F.relu(x)

            # Perform downsampling on last layer and add final residual
            residual = x
            x = b_l[-1]*self.final_block(x)

            if self.downsample:
                x += self.pad_identity(residual)
            else:
                # Don't downsample on final layer
                x += residual
            return F.relu(x)

    def pad_identity(self, x):
        # Perform padding on filters to allow final residual connections
        return (F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.out_filters//4, self.out_filters//4), "constant", 0))


# ----------StochasticDepthResNet----------

class StochasticDepthResNet(nn.Module):

    def __init__(self, filters_list, N, p_L=0.5):
        super(StochasticDepthResNet, self).__init__()
        """Stochastic depth ResNet as described in the paper
        'Deep Networks with Stochastic Depth'
        <https://arxiv.org/abs/1603.09382>

        Args:
            filters_list: List of different filter sizes for each reisudal block
                in the network. Default value in paper is [16, 32, 64].
            N: Int controlling the overall depth of our ResNet. Total
                number of layers is (6N+2).
            p_L: value representing the survival probability of the final layer,
                all other survival probabilities are a linear function of p_L. Default
            value in paper is 0.5

        """

        self.filters_list = filters_list
        self.N = N

        self.p_L = p_L

        # Linear decay method for probabilities
        self.layer_probs = torch.tensor(
            [1-((i/(6*self.N))*(1-self.p_L)) for i in range(1, (6*self.N)+1)])

        # Following paper, p_0 = 1

        self.layer_probs[0] = 1.0

        self.first_layer = nn.Conv2d(
            3, filters_list[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(filters_list[0])
        self.first_block = ResidualBlock(
            in_filters=self.filters_list[0], out_filters=self.filters_list[1], N=self.N, layer_probs=self.layer_probs[0:self.N])

        self.second_block = ResidualBlock(
            in_filters=self.filters_list[1], out_filters=self.filters_list[2], N=self.N, layer_probs=self.layer_probs[self.N:2*self.N])

        self.third_block = ResidualBlock(
            in_filters=self.filters_list[2], out_filters=self.filters_list[2], N=self.N, layer_probs=self.layer_probs[2*self.N:], downsample=False)
        self.fc = nn.Linear(64, 10)

        self.apply(_weights_init)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.bn(self.first_layer(x)))
        x = self.first_block(x)
        x = self.second_block(x)
        x = self.third_block(x)

        # Global average pooling
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':

    import torch.autograd.profiler as profiler
    model = StochasticDepthResNet(filters_list=[16, 32, 64], N=18)
    inputs = torch.randn(5, 3, 32, 32)

    with profiler.profile(record_shapes=True) as prof:
        with profiler.record_function("model_inference"):
            model(inputs)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
