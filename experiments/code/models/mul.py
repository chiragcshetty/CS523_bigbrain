"""
run_train: train the dm.parallelModelThreeLayerSplit model using given settings

*****MODEL*******
- dm.parallelModelThreeLayerSplit: Looks as below
           ----|---|----                       .....Branch 1
          /             \
     | --- ----|---|---------|-----|--(output) .....Branch 2
    (L1)  \             /  (L3)   (L4)
           ----|---|----                       .....Branch 3
           
- On one GPU it runs as GPU1: (L1)(Branch1)(Branch2)(Branch3)(L3)(L4)
- On 2 GPU's it runs as GPU1: (L1)(Branch1)(Branch2)(L3)(L4)
                        GPU2:     (Branch3)
                        
- 2 GPU execution is about 20% faster
- Even when provided 3 GPUs, fitting in 2 GPUs gives least step time

-Inputs to the model:dm.parallelModelThreeLayerSplit(factor, split, repetable)
    -factor : Factor by which all the layers will be scaled. eg: factor =2 will doubble size of all layers
    -split : A list of ints [g1, g2]: g1, g2 are device ids for GPU1, GPU2 above
        - Though not exact, for this exp we assume [gpu1,gpu1] occupies 0.6 of gpu1, [gpu1,gpu2] occupies 0.3 of each gpu1 and gpu2
    -repetable : When repetable is non-zero, the model weights are initialized to 1/512 and biases to 0
******************

run_train arguments:
- split, factor: As described above
- batch_size: For training (done using fake dataset)
- Nrun: Number of batches (steps)
- done_flag: A singleton list (so that it is mutable) to indicate end of thread (contains avg step time)
- exit_time: Time when job is completed
"""


import argparse
import gc
from random import sample

import torch
import torchvision
from torch import nn, optim

import models.parallel as dm
import time

parser = argparse.ArgumentParser()

parser.add_argument(
    "--split", help="GPU splitting", nargs=2, type=int, default=(0, 0)
)
parser.add_argument("--factor", help="Proxy for model size", type=float, default=6.0)
parser.add_argument("--batch_size", help="Batch size", type=int, default=64)
parser.add_argument("--samples", help="Number of batches", type=int, default=1000)

args = parser.parse_args()

split: tuple = args.split
factor: int = int(args.factor)
batch_size: int = args.batch_size
samples: int = args.samples

print(split)

######## For profiler (some experiments. Not required) #################
# from torch.profiler import profile, record_function, ProfilerActivit

tensor1 = torch.randn(batch_size, factor, factor).to(split[0])
tensor2 = torch.randn(batch_size, factor, factor).to(split[0])

start = time.time()
if __name__ == "__main__":

    for i in range(samples):
        out = torch.matmul(tensor1, tensor2)
#        print(i)
        
    end = time.time()
    print(end-start)
    # #### Release memory
    # del model
    # del inp
    # del output
    # try:
    #     del labels
    #     del optimizer
    #     del loss
    # except:
    #     pass
    # gc.collect()  ## To clean any circular references
    # torch.cuda.empty_cache()  ## Empty cache used by Pytorch (does so across all threads)
