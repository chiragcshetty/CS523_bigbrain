#############################################
## Copy of Inceptionv3, slightly modified for recording intermeridates
# sys.path.append('/home/cshetty2/sct/pytorch')
# import reformated_models.inception_modified as inception_modified

## Modified Alexnet, with a'factor' by which it can be made 'fat'
# import simple_model as sm

## Placer libs of baechi
# sys.path.append('/home/cshetty2/sct')
# from placer.placer_lib import *
##############################################

import torch
import torch.nn as nn
from typing import Any

import torch.nn.functional as F


class _concatenateLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *x):
        return torch.cat(x, 1)


class _squeezeLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.squeeze()


class _addLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return x1 + x2


class ParallelModel(nn.Module):
    def __init__(self, factor: float = 1.0) -> None:
        super().__init__()  # python 3 syntax

        self.factor = factor
        self.linear1N = int(512 * self.factor)
        self.linear2N = int(2048 * self.factor)
        self.linear3N = int(512 * self.factor)
        self.linear4N = 2 * self.linear3N
        self.linear5N = int(512 * self.factor)

        self.squeeze = _squeezeLayer()
        self.fc1 = nn.Linear(self.linear1N, self.linear2N)
        self.fc2a1 = nn.Linear(self.linear2N, self.linear3N)
        self.fc2a2 = nn.Linear(self.linear3N, self.linear3N)
        self.fc2b1 = nn.Linear(self.linear2N, self.linear3N)
        self.fc2b2 = nn.Linear(self.linear3N, self.linear3N)
        self.concatenate = _concatenateLayer()
        self.fc3 = nn.Linear(self.linear4N, self.linear5N)
        # self.add1 = _addLayer()
        self.fc4 = nn.Linear(self.linear5N, self.linear5N)

    def forward(self, x):
        x = self.squeeze(x)
        x = self.fc1(x)
        xa1 = self.fc2a1(x)
        xa2 = self.fc2a2(xa1)
        xb1 = self.fc2b1(x)
        xb2 = self.fc2b2(xb1)
        y = self.concatenate(xa2, xb2)
        y = self.fc3(y)
        # y = self.add1(y,xb2)
        y = self.fc4(y)
        return y


class ParallelModelSplit(nn.Module):
    def __init__(self, factor, layers, repetable=0) -> None:
        super().__init__()  # python 3 syntax

        self.factor = factor
        self.layers = layers
        self.linear1N = 512 * self.factor
        self.linear2N = 2048 * self.factor
        self.linear3N = 512 * self.factor
        self.linear4N = 2 * self.linear3N
        self.linear5N = 512 * self.factor

        self.squeeze = _squeezeLayer().to(self.layers[0])
        self.fc1 = nn.Linear(self.linear1N, self.linear2N).to(self.layers[0])
        self.fc2a1 = nn.Linear(self.linear2N, self.linear3N).to(self.layers[1])
        self.fc2a2 = nn.Linear(self.linear3N, self.linear3N).to(self.layers[1])
        self.fc2b1 = nn.Linear(self.linear2N, self.linear3N).to(self.layers[0])
        self.fc2b2 = nn.Linear(self.linear3N, self.linear3N).to(self.layers[0])
        self.concatenate = _concatenateLayer().to(self.layers[1])
        self.fc3 = nn.Linear(self.linear4N, self.linear5N).to(self.layers[1])
        # self.add1 = _addLayer()
        self.fc4 = nn.Linear(self.linear5N, self.linear5N).to(self.layers[1])

        if repetable:
            torch.nn.init.constant_(self.fc1.weight, 1 / 512)
            torch.nn.init.zeros_(self.fc1.bias)
            torch.nn.init.constant_(self.fc2a1.weight, 1 / 512)
            torch.nn.init.zeros_(self.fc2a1.bias)
            torch.nn.init.constant_(self.fc2a2.weight, 1 / 512)
            torch.nn.init.zeros_(self.fc2a2.bias)
            torch.nn.init.constant_(self.fc2b1.weight, 1 / 512)
            torch.nn.init.zeros_(self.fc2b1.bias)
            torch.nn.init.constant_(self.fc2b2.weight, 1 / 512)
            torch.nn.init.zeros_(self.fc2b2.bias)
            torch.nn.init.constant_(self.fc3.weight, 1 / 512)
            torch.nn.init.zeros_(self.fc3.bias)
            torch.nn.init.constant_(self.fc4.weight, 1 / 512)
            torch.nn.init.zeros_(self.fc4.bias)

    def forward(self, x):
        x = self.squeeze(x)
        x = self.fc1(x)

        x_t = x.to(self.layers[1])
        xa1 = self.fc2a1(x_t)
        xa2 = self.fc2a2(xa1)

        xb1 = self.fc2b1(x)
        xb2 = self.fc2b2(xb1)

        xb2_t = xb2.to(self.layers[1])
        y = self.concatenate(xa2, xb2_t)
        y = self.fc3(y)
        # y = self.add1(y,xb2)
        y = self.fc4(y)
        return y


class ParallelModelThreeLayer(nn.Module):
    def __init__(self, factor: int = 1) -> None:
        super().__init__()  # python 3 syntax

        self.factor = factor
        self.linear1N = 512 * self.factor
        self.linear2N = 2048 * self.factor
        self.linear3N = 1024 * self.factor
        self.linear4N = 3 * self.linear3N
        self.linear5N = 512 * self.factor

        self.squeeze = _squeezeLayer()
        self.fc1 = nn.Linear(self.linear1N, self.linear2N)
        self.fc2a1 = nn.Linear(self.linear2N, self.linear3N)
        self.fc2a2 = nn.Linear(self.linear3N, self.linear3N)
        self.fc2b1 = nn.Linear(self.linear2N, self.linear3N)
        self.fc2b2 = nn.Linear(self.linear3N, self.linear3N)
        self.fc2c1 = nn.Linear(self.linear2N, self.linear3N)
        self.fc2c2 = nn.Linear(self.linear3N, self.linear3N)
        self.concatenate = _concatenateLayer()
        self.fc3 = nn.Linear(self.linear4N, self.linear5N)
        self.fc4 = nn.Linear(self.linear5N, self.linear5N)

    def forward(self, x):
        x = self.squeeze(x)
        x = self.fc1(x)
        xa1 = self.fc2a1(x)
        xa2 = self.fc2a2(xa1)
        xb1 = self.fc2b1(x)
        xb2 = self.fc2b2(xb1)
        xc1 = self.fc2c1(x)
        xc2 = self.fc2c2(xc1)
        y = self.concatenate(xa2, xb2, xc2)
        y = self.fc3(y)
        y = self.fc4(y)
        return y


class ParallelModelThreeLayerSplit(nn.Module):
    def __init__(self, factor, layers, repetable=0) -> None:
        super().__init__()  # python 3 syntax

        self.factor = factor
        self.layers = layers

        self.linear1N =int( 512 * self.factor)
        self.linear2N = int(2048 * self.factor)
        self.linear3N = int(1024 * self.factor)
        self.linear4N = 3 * self.linear3N
        self.linear5N = int(512 * self.factor)

        self.squeeze = _squeezeLayer().to(layers[0])
        self.fc1 = nn.Linear(self.linear1N, self.linear2N).to(layers[0])
        self.fc2a1 = nn.Linear(self.linear2N, self.linear3N).to(layers[0])
        self.fc2a2 = nn.Linear(self.linear3N, self.linear3N).to(layers[0])
        self.fc2b1 = nn.Linear(self.linear2N, self.linear3N).to(layers[1])
        self.fc2b2 = nn.Linear(self.linear3N, self.linear3N).to(layers[1])
        self.fc2c1 = nn.Linear(self.linear2N, self.linear3N).to(layers[0])
        self.fc2c2 = nn.Linear(self.linear3N, self.linear3N).to(layers[0])
        self.concatenate = _concatenateLayer().to(layers[0])
        self.fc3 = nn.Linear(self.linear4N, self.linear5N).to(layers[0])
        self.fc4 = nn.Linear(self.linear5N, self.linear5N).to(layers[0])

        if repetable:
            torch.nn.init.constant_(self.fc1.weight, 1 / 512)
            torch.nn.init.zeros_(self.fc1.bias)
            torch.nn.init.constant_(self.fc2a1.weight, 1 / 512)
            torch.nn.init.zeros_(self.fc2a1.bias)
            torch.nn.init.constant_(self.fc2a2.weight, 1 / 512)
            torch.nn.init.zeros_(self.fc2a2.bias)
            torch.nn.init.constant_(self.fc2b1.weight, 1 / 512)
            torch.nn.init.zeros_(self.fc2b1.bias)
            torch.nn.init.constant_(self.fc2b2.weight, 1 / 512)
            torch.nn.init.zeros_(self.fc2b2.bias)
            torch.nn.init.constant_(self.fc2c1.weight, 1 / 512)
            torch.nn.init.zeros_(self.fc2c1.bias)
            torch.nn.init.constant_(self.fc2c2.weight, 1 / 512)
            torch.nn.init.zeros_(self.fc2c2.bias)
            torch.nn.init.constant_(self.fc3.weight, 1 / 512)
            torch.nn.init.zeros_(self.fc3.bias)
            torch.nn.init.constant_(self.fc4.weight, 1 / 512)
            torch.nn.init.zeros_(self.fc4.bias)

    def forward(self, x):
        x = x.to(self.layers[0])
        x = self.squeeze(x)
        x = self.fc1(x)

        xa1 = self.fc2a1(x)
        xa2 = self.fc2a2(xa1)

        x_t = x.to(self.layers[1])
        xb1 = self.fc2b1(x_t)
        xb2 = self.fc2b2(xb1)

        xc1 = self.fc2c1(x)
        xc2 = self.fc2c2(xc1)

        xb2_t = xb2.to(self.layers[0])

        y = self.concatenate(xa2, xb2_t, xc2)
        y = self.fc3(y)
        y = self.fc4(y)
        return y


class ParallelModelThreeLayerSplitOnes(nn.Module):
    def __init__(self, factor, layers) -> None:
        super().__init__()  # python 3 syntax

        self.factor = factor
        self.layers = layers

        self.linear1N = 512 * self.factor
        self.linear2N = 2048 * self.factor
        self.linear3N = 1024 * self.factor
        self.linear4N = 3 * self.linear3N
        self.linear5N = 512 * self.factor

        self.squeeze = _squeezeLayer().to(layers[0])
        self.fc1 = nn.Linear(self.linear1N, self.linear2N).to(layers[0])
        self.fc2a1 = nn.Linear(self.linear2N, self.linear3N).to(layers[0])
        self.fc2a2 = nn.Linear(self.linear3N, self.linear3N).to(layers[0])
        self.fc2b1 = nn.Linear(self.linear2N, self.linear3N).to(layers[1])
        self.fc2b2 = nn.Linear(self.linear3N, self.linear3N).to(layers[1])
        self.fc2c1 = nn.Linear(self.linear2N, self.linear3N).to(layers[0])
        self.fc2c2 = nn.Linear(self.linear3N, self.linear3N).to(layers[0])
        self.concatenate = _concatenateLayer().to(layers[0])
        self.fc3 = nn.Linear(self.linear4N, self.linear5N).to(layers[0])
        self.fc4 = nn.Linear(self.linear5N, self.linear5N).to(layers[0])

    def forward(self, x):
        x = self.squeeze(x)
        x = self.fc1(x)

        xa1 = self.fc2a1(x)
        xa2 = self.fc2a2(xa1)

        x_t = x.to(self.layers[1])
        xb1 = self.fc2b1(x_t)
        xb2 = self.fc2b2(xb1)

        xc1 = self.fc2c1(x)
        xc2 = self.fc2c2(xc1)

        xb2_t = xb2.to(self.layers[0])

        y = self.concatenate(xa2, xb2_t, xc2)
        y = self.fc3(y)
        y = self.fc4(y)
        return y


def parallelModel(factor) -> ParallelModel:
    model = ParallelModel(factor)
    return model


def parallelModelSplit(factor, layers, repetable=0) -> ParallelModelSplit:
    model = ParallelModelSplit(factor, layers, repetable)
    return model


def parallelModelThreeLayer(factor) -> ParallelModelThreeLayer:
    model = ParallelModelThreeLayer(factor)
    return model


def parallelModelThreeLayerSplit(
    factor, layers, repetable=0
) -> ParallelModelThreeLayerSplit:
    model = ParallelModelThreeLayerSplit(factor, layers, repetable)
    return model
