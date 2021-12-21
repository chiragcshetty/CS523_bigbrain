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


class ParallelTwoLayer(nn.Module):

    def __init__(self, factor, repetable =0) -> None:
        super().__init__() # python 3 syntax
        
        self.factor = factor
        self.linear1N = int(512*self.factor)
        self.linear2N = int(2048*self.factor)
        self.linear3N = int(512*self.factor)
        self.linear4N = 2*self.linear3N
        self.linear5N = int(512*self.factor)


        self.squeeze = _squeezeLayer()
        self.fc1 = nn.Linear(self.linear1N, self.linear2N)
        self.fc2a1 = nn.Linear(self.linear2N, self.linear3N)
        self.fc2a2 = nn.Linear(self.linear3N, self.linear3N)
        self.fc2b1 = nn.Linear(self.linear2N, self.linear3N)
        self.fc2b2 = nn.Linear(self.linear3N, self.linear3N)
        self.concatenate = _concatenateLayer()
        self.fc3 = nn.Linear(self.linear4N, self.linear5N)
        #self.add1 = _addLayer()
        self.fc4 = nn.Linear(self.linear5N, self.linear5N)

        if repetable:
            torch.nn.init.constant_(self.fc1.weight, 1/512); torch.nn.init.zeros_(self.fc1.bias)
            torch.nn.init.constant_(self.fc2a1.weight, 1/512); torch.nn.init.zeros_(self.fc2a1.bias)
            torch.nn.init.constant_(self.fc2a2.weight, 1/512); torch.nn.init.zeros_(self.fc2a2.bias)
            torch.nn.init.constant_(self.fc2b1.weight, 1/512); torch.nn.init.zeros_(self.fc2b1.bias)
            torch.nn.init.constant_(self.fc2b2.weight, 1/512); torch.nn.init.zeros_(self.fc2b2.bias)
            torch.nn.init.constant_(self.fc3.weight, 1/512); torch.nn.init.zeros_(self.fc3.bias)
            torch.nn.init.constant_(self.fc4.weight, 1/512); torch.nn.init.zeros_(self.fc4.bias)
        

    def forward(self, x):
        x = self.squeeze(x)
        x = self.fc1(x)
        xa1 = self.fc2a1(x)
        xa2 = self.fc2a2(xa1)
        xb1 = self.fc2b1(x)
        xb2 = self.fc2b2(xb1)
        y = self.concatenate(xa2,xb2)
        y = self.fc3(y)
        #y = self.add1(y,xb2)
        y = self.fc4(y)
        return y


class ParallelThreeLayer(nn.Module):

    def __init__(self, factor, repetable =0) -> None:
        super().__init__() # python 3 syntax
        
        self.factor = factor
        self.linear1N = int(512*self.factor)
        self.linear2N = int(2048*self.factor)
        self.linear3N = int(512*self.factor)
        #self.linear3N = int(1024*self.factor)
        self.linear4N = 3*self.linear3N
        self.linear5N = int(512*self.factor)


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
        #self.add1 = _addLayer()
        self.fc4 = nn.Linear(self.linear5N, self.linear5N)

        if repetable:
            torch.nn.init.constant_(self.fc1.weight, 1/512); torch.nn.init.zeros_(self.fc1.bias)
            torch.nn.init.constant_(self.fc2a1.weight, 1/512); torch.nn.init.zeros_(self.fc2a1.bias)
            torch.nn.init.constant_(self.fc2a2.weight, 1/512); torch.nn.init.zeros_(self.fc2a2.bias)
            torch.nn.init.constant_(self.fc2b1.weight, 1/512); torch.nn.init.zeros_(self.fc2b1.bias)
            torch.nn.init.constant_(self.fc2b2.weight, 1/512); torch.nn.init.zeros_(self.fc2b2.bias)
            torch.nn.init.constant_(self.fc2c1.weight, 1/512); torch.nn.init.zeros_(self.fc2c1.bias)
            torch.nn.init.constant_(self.fc2c2.weight, 1/512); torch.nn.init.zeros_(self.fc2c2.bias)
            torch.nn.init.constant_(self.fc3.weight, 1/512); torch.nn.init.zeros_(self.fc3.bias)
            torch.nn.init.constant_(self.fc4.weight, 1/512); torch.nn.init.zeros_(self.fc4.bias)
        

    def forward(self, x):
        x = self.squeeze(x)
        x = self.fc1(x)
        xa1 = self.fc2a1(x)
        xa2 = self.fc2a2(xa1)
        xb1 = self.fc2b1(x)
        xb2 = self.fc2b2(xb1)
        xc1 = self.fc2c1(x)
        xc2 = self.fc2c2(xc1)
        y = self.concatenate(xa2,xb2,xc2)
        y = self.fc3(y)
        #y = self.add1(y,xb2)
        y = self.fc4(y)
        return y

class ParallelThreeLayerOld(nn.Module):

    def __init__(self, factor: int = 1) -> None:
        super().__init__() # python 3 syntax
        
        self.factor = factor
        self.linear1N = 512*self.factor
        self.linear2N = 2048*self.factor
        self.linear3N = 1024*self.factor
        self.linear4N = 3*self.linear3N
        self.linear5N = 512*self.factor


        self.squeeze = _squeezeLayer()
        
        self.fc1   = nn.Linear(self.linear1N, self.linear2N)
        torch.nn.init.constant_(self.fc1.weight, 1/512)
        torch.nn.init.zeros_(self.fc1.bias)
        
        self.fc2a1 = nn.Linear(self.linear2N, self.linear3N)
        torch.nn.init.constant_(self.fc2a1.weight, 1/512)
        torch.nn.init.zeros_(self.fc2a1.bias)
        
        self.fc2a2 = nn.Linear(self.linear3N, self.linear3N)
        torch.nn.init.constant_(self.fc2a2.weight, 1/512)
        torch.nn.init.zeros_(self.fc2a2.bias)
        
        self.fc2b1 = nn.Linear(self.linear2N, self.linear3N)
        torch.nn.init.constant_(self.fc2b1.weight, 1/512)
        torch.nn.init.zeros_(self.fc2b1.bias)
        
        self.fc2b2 = nn.Linear(self.linear3N, self.linear3N)
        torch.nn.init.constant_(self.fc2b2.weight, 1/512)
        torch.nn.init.zeros_(self.fc2b2.bias)
        
        self.fc2c1 = nn.Linear(self.linear2N, self.linear3N)
        torch.nn.init.constant_(self.fc2c1.weight, 1/512)
        torch.nn.init.zeros_(self.fc2c1.bias)
        
        self.fc2c2 = nn.Linear(self.linear3N, self.linear3N)
        torch.nn.init.constant_(self.fc2c2.weight, 1/512)
        torch.nn.init.zeros_(self.fc2c2.bias)
        
        self.concatenate = _concatenateLayer()
        
        self.fc3   = nn.Linear(self.linear4N, self.linear5N)
        torch.nn.init.constant_(self.fc3.weight, 1/512)
        torch.nn.init.zeros_(self.fc3.bias)
        
        
        self.fc4   = nn.Linear(self.linear5N, self.linear5N)
        torch.nn.init.constant_(self.fc4.weight, 1/512)
        torch.nn.init.zeros_(self.fc4.bias)
        

    def forward(self, x):
        x = self.squeeze(x)
        x = self.fc1(x)
        xa1 = self.fc2a1(x)
        xa2 = self.fc2a2(xa1)
        xb1 = self.fc2b1(x)
        xb2 = self.fc2b2(xb1)
        xc1 = self.fc2c1(x)
        xc2 = self.fc2c2(xc1)
        y = self.concatenate(xa2,xb2,xc2)
        y = self.fc3(y)
        y1 = self.fc4(y)
        return y1


class TallParallelModel(nn.Module):

    def __init__(self, factor, repetable =0) -> None:
        super().__init__() # python 3 syntax
        
        self.factor = factor
        self.linear1N = int(512*self.factor)
        self.linear2N = int(2048*self.factor)
        self.linear3N = int(512*self.factor)
        self.linear4N = 2*self.linear3N
        self.linear5N = int(512*self.factor)

        self.squeeze = _squeezeLayer()
        self.fc1 = nn.Linear(self.linear1N, self.linear2N)
        self.fc2a = nn.Linear(self.linear2N, self.linear3N)
        self.fc2b = nn.Linear(self.linear2N, self.linear3N)
        self.concatenate = _concatenateLayer()
        self.fc3 = nn.Linear(self.linear4N, self.linear5N)
        self.fc4 = nn.Linear(self.linear5N, self.linear5N)
        self.fc5 = nn.Linear(self.linear5N, self.linear5N)
        self.fc6 = nn.Linear(self.linear5N, self.linear5N)
        self.fc7 = nn.Linear(self.linear5N, self.linear5N)
        self.fc8 = nn.Linear(self.linear5N, self.linear5N)
        self.fc9 = nn.Linear(self.linear5N, self.linear5N)
        self.fc10 = nn.Linear(self.linear5N, self.linear5N)
        self.fc11 = nn.Linear(self.linear5N, self.linear5N)
        self.fc12 = nn.Linear(self.linear5N, self.linear5N)
        self.fc13 = nn.Linear(self.linear5N, self.linear5N)
        self.fc14 = nn.Linear(self.linear5N, self.linear5N)
        self.fc15 = nn.Linear(self.linear5N, self.linear5N)
        self.fc16 = nn.Linear(self.linear5N, self.linear5N)
        self.fc17 = nn.Linear(self.linear5N, self.linear5N)
        self.fc18 = nn.Linear(self.linear5N, self.linear5N)
        self.fc19 = nn.Linear(self.linear5N, self.linear5N)
        self.fc20 = nn.Linear(self.linear5N, self.linear5N)
        self.fc21 = nn.Linear(self.linear5N, self.linear5N)
        self.fc22 = nn.Linear(self.linear5N, self.linear5N)
        self.fc23 = nn.Linear(self.linear5N, self.linear5N)
        self.fc24 = nn.Linear(self.linear5N, self.linear5N)
        self.fc25 = nn.Linear(self.linear5N, self.linear5N)
        self.fc26 = nn.Linear(self.linear5N, self.linear5N)
        self.fc27 = nn.Linear(self.linear5N, self.linear5N)
        self.fc28 = nn.Linear(self.linear5N, self.linear5N)
        self.fc29 = nn.Linear(self.linear5N, self.linear5N)

        if repetable:
            torch.nn.init.constant_(self.fc1.weight, 1/512); torch.nn.init.zeros_(self.fc1.bias)
            torch.nn.init.constant_(self.fc2a.weight, 1/512); torch.nn.init.zeros_(self.fc2a.bias)
            torch.nn.init.constant_(self.fc2b.weight, 1/512); torch.nn.init.zeros_(self.fc2b.bias)
            torch.nn.init.constant_(self.fc3.weight, 1/512); torch.nn.init.zeros_(self.fc3.bias)
            torch.nn.init.constant_(self.fc4.weight, 1/512); torch.nn.init.zeros_(self.fc4.bias)
            torch.nn.init.constant_(self.fc5.weight, 1/512); torch.nn.init.zeros_(self.fc5.bias)
            torch.nn.init.constant_(self.fc6.weight, 1/512); torch.nn.init.zeros_(self.fc6.bias)
            torch.nn.init.constant_(self.fc7.weight, 1/512); torch.nn.init.zeros_(self.fc7.bias)
            torch.nn.init.constant_(self.fc8.weight, 1/512); torch.nn.init.zeros_(self.fc8.bias)
            torch.nn.init.constant_(self.fc9.weight, 1/512); torch.nn.init.zeros_(self.fc9.bias)
            torch.nn.init.constant_(self.fc10.weight, 1/512); torch.nn.init.zeros_(self.fc10.bias)
            torch.nn.init.constant_(self.fc11.weight, 1/512); torch.nn.init.zeros_(self.fc11.bias)
            torch.nn.init.constant_(self.fc12.weight, 1/512); torch.nn.init.zeros_(self.fc12.bias)
            torch.nn.init.constant_(self.fc13.weight, 1/512); torch.nn.init.zeros_(self.fc13.bias)
            torch.nn.init.constant_(self.fc14.weight, 1/512); torch.nn.init.zeros_(self.fc14.bias)
            torch.nn.init.constant_(self.fc15.weight, 1/512); torch.nn.init.zeros_(self.fc15.bias)
            torch.nn.init.constant_(self.fc16.weight, 1/512); torch.nn.init.zeros_(self.fc16.bias)
            torch.nn.init.constant_(self.fc17.weight, 1/512); torch.nn.init.zeros_(self.fc17.bias)
            torch.nn.init.constant_(self.fc18.weight, 1/512); torch.nn.init.zeros_(self.fc18.bias)
            torch.nn.init.constant_(self.fc19.weight, 1/512); torch.nn.init.zeros_(self.fc19.bias)
            torch.nn.init.constant_(self.fc20.weight, 1/512); torch.nn.init.zeros_(self.fc20.bias)
            torch.nn.init.constant_(self.fc21.weight, 1/512); torch.nn.init.zeros_(self.fc21.bias)
            torch.nn.init.constant_(self.fc22.weight, 1/512); torch.nn.init.zeros_(self.fc22.bias)
            torch.nn.init.constant_(self.fc23.weight, 1/512); torch.nn.init.zeros_(self.fc23.bias)
            torch.nn.init.constant_(self.fc24.weight, 1/512); torch.nn.init.zeros_(self.fc24.bias)
            torch.nn.init.constant_(self.fc25.weight, 1/512); torch.nn.init.zeros_(self.fc25.bias)
            torch.nn.init.constant_(self.fc26.weight, 1/512); torch.nn.init.zeros_(self.fc26.bias)
            torch.nn.init.constant_(self.fc27.weight, 1/512); torch.nn.init.zeros_(self.fc27.bias)
            torch.nn.init.constant_(self.fc28.weight, 1/512); torch.nn.init.zeros_(self.fc28.bias)
            torch.nn.init.constant_(self.fc29.weight, 1/512); torch.nn.init.zeros_(self.fc29.bias)

          

    def forward(self, x):
        x = self.squeeze(x)
        x = self.fc1(x)
        xb = self.fc2b(x)
        xa = self.fc2a(x)
        y = self.concatenate(xa,xb)
        y = self.fc3(y)
        y = self.fc4(y)
        y = self.fc5(y)
        y = self.fc6(y)
        y = self.fc7(y)
        y = self.fc8(y)
        y = self.fc9(y)
        y = self.fc10(y)
        y = self.fc11(y)
        y = self.fc12(y)
        y = self.fc13(y)
        y = self.fc14(y)
        y = self.fc15(y)
        y = self.fc16(y)
        y = self.fc17(y)
        y = self.fc18(y)
        y = self.fc19(y)
        y = self.fc20(y)
        y = self.fc21(y)
        y = self.fc22(y)
        y = self.fc23(y)
        y = self.fc24(y)
        y = self.fc25(y)
        y = self.fc26(y)
        y = self.fc27(y)
        y = self.fc28(y)
        y = self.fc29(y)
        
        return y


def parallelThreeLayer(factor, repetable=0) -> ParallelThreeLayer:
    model = ParallelThreeLayer(factor, repetable)
    return model

def parallelTwoLayer(factor, repetable=0) -> ParallelTwoLayer:
    model = ParallelTwoLayer(factor, repetable)
    return model

def parallelThreeLayerOld(factor) -> ParallelThreeLayerOld:
    model = ParallelThreeLayerOld(factor)
    return model

def tallParallelModel(factor, repetable=0) -> TallParallelModel:
    model = TallParallelModel(factor, repetable)
    return model