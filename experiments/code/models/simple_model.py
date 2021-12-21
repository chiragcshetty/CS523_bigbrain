'''
Original Source: https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
Modified by Chirag C. Shetty (cshetty2@illinois.edu) 
Date: 19 July 2021
'''
import torch
import torch.nn as nn
from typing import Any

import torch.nn.functional as F


__all__ = ['SimpleModel', 'simpleModel']


'''
A simple CNN model, derived from alexnet, that can be made fat using the 'factor'
'''
class SimpleModel(nn.Module):

    def __init__(self, num_classes: int = 1000, factor: int = 1) -> None:
        super(SimpleModel, self).__init__()
        self.factor = factor

        ## Output channel at each conv layer
        self.conv1 = round(64*self.factor)
        self.conv2 = round(192*self.factor)
        self.conv3 = round(384*self.factor)
        self.conv4 = round(256*self.factor)
        self.conv5 = round(256*self.factor)
        self.avgpoolsize = round(6*self.factor)
        self.linear1 = round(4096*self.factor)
        self.linear2 = round(4096*self.factor)
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(3, self.conv1, kernel_size=11, stride=4, padding=2),
            ## nn.ReLU(inplace=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(self.conv1, self.conv2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(self.conv2, self.conv3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.conv3, self.conv4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.conv4, self.conv5, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((self.avgpoolsize, self.avgpoolsize))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.conv5 * self.avgpoolsize * self.avgpoolsize,  self.linear1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.linear1, self.linear2),
            nn.ReLU(),
            nn.Linear(self.linear2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

'''
Toy Model with 2 conv2d and 2 linear layers
'''
class ToyModel(nn.Module):

    def __init__(self, num_classes: int = 1000, factor: int = 1) -> None:
        super(ToyModel, self).__init__()
        self.factor = factor

        ## Output channel at each conv layer
        self.conv1 = round(64*self.factor)
        self.conv2 = round(192*self.factor)

        self.avgpoolsize = round(6*self.factor)

        self.linear1 = round(4096*self.factor)
        self.linear2 = round(4096*self.factor)

        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(3, self.conv1, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(self.conv1, self.conv2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((self.avgpoolsize, self.avgpoolsize))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.conv2 * self.avgpoolsize * self.avgpoolsize,  self.linear1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.linear1, self.linear2),
            nn.ReLU(),
            nn.Linear(self.linear2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

'''
Toy Model with 2 conv2d and 2 linear layers and no categories within them
Modified from CS498 Assignment on Alexnet
'''
class ToyToyModel(nn.Module):

    def __init__(self, num_classes: int = 1000, factor: int = 1) -> None:
        super(ToyToyModel, self).__init__()
        self.factor = factor

        ## Output channel at each conv layer
        self.conv1N = round(64*self.factor)
        #self.conv2N = round(192*self.factor)
        self.avgpoolsizeN = 3
        self.linear1N = round(4096*self.factor)
        self.linear2N = round(4096*self.factor)

        self.num_classes = num_classes

        self.features = nn.Sequential(
            #nn.Conv2d(3, self.conv1N, 5),
            nn.Conv2d(3, self.conv1N, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            )
        self.avgpool = nn.AdaptiveAvgPool2d((self.avgpoolsizeN,self.avgpoolsizeN))
        self.classifier = nn.Sequential(
            nn.Linear(self.conv1N * self.avgpoolsizeN * self.avgpoolsizeN,  self.linear1N),
            nn.Linear(self.linear1N, self.linear2N),
            nn.Linear(self.linear2N, num_classes),
            )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


'''
Copy of ToyToyModel with two parallel linear layers at the end
'''
class _concatenateLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *x):
        return torch.cat(x, 1)

class ParallelToyModel(nn.Module):

    def __init__(self, num_classes: int = 1000, factor: int = 1) -> None:
        super(ParallelToyModel, self).__init__()
        self.factor = factor

        ## Output channel at each conv layer
        self.conv1N = round(64*self.factor)
        #self.conv2N = round(192*self.factor)
        self.avgpoolsizeN = 3
        self.linear1N = round(4096*self.factor)
        self.linear2N = round(4096*self.factor)

        self.num_classes = num_classes

        self.features = nn.Sequential(
            #nn.Conv2d(3, self.conv1N, 5),
            nn.Conv2d(3, self.conv1N, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            )
        self.avgpool = nn.AdaptiveAvgPool2d((self.avgpoolsizeN,self.avgpoolsizeN))
        self.classifier1 = nn.Sequential(
            nn.Linear(self.conv1N * self.avgpoolsizeN * self.avgpoolsizeN,  self.linear1N),
            nn.Linear(self.linear1N, self.linear2N),
            nn.Linear(self.linear2N, int(num_classes/2)),
            )
        self.classifier2 = nn.Sequential(
            nn.Linear(self.conv1N * self.avgpoolsizeN * self.avgpoolsizeN,  self.linear1N),
            nn.Linear(self.linear1N, self.linear2N),
            nn.Linear(self.linear2N, int(num_classes/2)),
            )
        self.concatenateFinal = _concatenateLayer()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x1 = self.classifier1(x)
        x2 = self.classifier2(x)
        y = self.concatenateFinal(x1,x2)
        return y


"LinearModel"
class LinearModel(nn.Module):

    def __init__(self, num_classes: int = 1000, factor: int = 1) -> None:
        super(LinearModel, self).__init__()
        self.factor = factor
        self.linear1N = 4096*self.factor
        self.linear2N = 4096*self.factor
        self.linear3N = 4096*self.factor


        self.fc1 = nn.Linear(10000,  self.linear1N)
        self.rl1 = nn.ReLU()
        self.fc2 = nn.Linear(self.linear1N, self.linear2N)
        self.rl2 = nn.ReLU()
        self.fc3 = nn.Linear(self.linear2N, self.linear3N)
        self.rl3 = nn.ReLU()
        self.fc4 = nn.Linear(self.linear3N, num_classes)

    def forward(self, x):

        x = self.fc1(x)
        x = self.rl1(x)

        x = self.fc2(x)
        x = self.rl2(x)

        x = self.fc3(x)
        x = self.rl3(x)

        x = self.fc4(x)

        return x


"Two layer LinearModel"
class TwoLayerLinearModel(nn.Module):

    def __init__(self, factor: int = 1) -> None:
        super(TwoLayerLinearModel, self).__init__()
        self.factor = factor
        self.linear1N = 512*self.factor
        self.linear2N = 2048*self.factor
        self.linear3N = 1024*self.factor


        self.fc1 = nn.Linear(self.linear1N, self.linear2N)
        self.fc2 = nn.Linear(self.linear2N, self.linear3N)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


'''
Model with a small number of parameters to verify tesnor sizes etc
'''
class SmallModel(nn.Module):

    def __init__(self, factor: int = 1) -> None:
        super(SmallModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 2, kernel_size=2),
            nn.Conv2d(2, 2, kernel_size=2),
            )
        self.avgpool = nn.AdaptiveAvgPool2d((3,3))
        self.classifier1 = nn.Sequential(
            nn.Linear(2*3*3,  3),
            nn.Linear(3, 2),
            )
        self.classifier2 = nn.Sequential(
            nn.Linear(2*3*3,  3),
            nn.Linear(3, 2),
            )
        self.concatenateFinal = _concatenateLayer()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x1 = self.classifier1(x)
        x2 = self.classifier2(x)
        y = self.concatenateFinal(x1,x2)
        return y



def simpleModel(progress: bool = True, **kwargs: Any) -> SimpleModel:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    The required minimum input size of the model is 63x63.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = SimpleModel(**kwargs)
    return model


def toyModel(progress: bool = True, **kwargs: Any) -> ToyModel:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    The required minimum input size of the model is 63x63.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ToyModel(**kwargs)
    return model

def toyToyModel(progress: bool = True, **kwargs: Any) -> ToyToyModel:
    model = ToyToyModel(**kwargs)
    return model

def linearModel(progress: bool = True, **kwargs: Any) -> LinearModel:
    model = LinearModel(**kwargs)
    return model

def parallelToyModel(progress: bool = True, **kwargs: Any) -> ParallelToyModel:
    model = ParallelToyModel(**kwargs)
    return model

def smallModel(progress: bool = True, **kwargs: Any) -> SmallModel:
    model = SmallModel(**kwargs)
    return model

def twoLayerLinearModel(progress: bool = True, **kwargs: Any) -> TwoLayerLinearModel:
    model = TwoLayerLinearModel(**kwargs)
    return model