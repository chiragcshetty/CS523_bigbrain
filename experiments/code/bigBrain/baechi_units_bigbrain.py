import torch
import torchvision
from torchvision import models
import time
import networkx as nx
from torch import optim, nn
from importlib import reload
import numpy as np
import pickle

import GPUtil

import sys


## Copy of Inceptionv3, slightly modified for recording intermeridates
sys.path.append('/home/cshetty2/sct/pytorch')
import reformated_models.inception_modified as inception_modified

## Modified Alexnet, with a'factor' by which it can be made 'fat' 
import simple_model as sm

######## For profiler (some experiments. Not required) #################
from torch.profiler import profile, record_function, ProfilerActivity


## Placer libs of baechi
sys.path.append('/home/cshetty2/sct')
from placer.placer_lib import *

import matplotlib.pyplot as plt


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Defined in this round about way (instead of just directly assigning) to keep it compatibble with summarize.py
class Args:
     def __init__(self,itype, prof_rounds, prof_gpu_id, batch_size, gpu_num, sch):
         self.type = itype
         self.prof_rounds = prof_rounds
         self.prof_gpu_id = prof_gpu_id
         self.batch_size = batch_size
         self.gpu_num = gpu_num
         self.sch = sch
            
itype       = 'forward'  # help: forward/all -> Conside forward path only or both
prof_rounds = 10      # help: 'rounds for profiler'
prof_gpu_id = 3      # help: 'which gpu to place the profiler'
batch_size  = '32'   # help: 'batch_size'
gpu_num     = 3      # help: 'number of gpu to use'
sch         = 'sct'  # help: 'sct/etf/topo'

args = Args(itype, prof_rounds, prof_gpu_id, batch_size, gpu_num, sch)
##########################################################################

"""
    Function: placer_lib.create_device_graph
    -> Creates a graph with devices as nodes and unit weight edges between them
    -> Each node: graph.add_node(device_id,
                                 id=device_id,
                                 name=device_info["name"],
                                 size=0,
                                 memory_limit=device_info["memory_size"])
"""
DEVICE_GRAPH_SINGLE = create_device_graph({0: {'name': '/job:localhost/replica:0/task:0/device:XLA_GPU:0', 'memory_size':  17179869184, 'type': ''}})
DEVICE_GRAPH_MULTIPLE = create_device_graph({0: {'name': '/job:localhost/replica:0/task:0/device:XLA_GPU:0', 'memory_size': 8000000000, 'type': ''}, 
                                             1: {'name': '/job:localhost/replica:0/task:0/device:XLA_GPU:1', 'memory_size': 8000000000, 'type': ''}, 
                                             2: {'name': '/job:localhost/replica:0/task:0/device:XLA_GPU:2', 'memory_size': 8000000000, 'type': ''}, 
                                             3: {'name': '/job:localhost/replica:0/task:0/device:XLA_GPU:3', 'memory_size': 8000000000, 'type': ''}})


device_list = {0: {'name': '/job:localhost/replica:0/task:0/device:XLA_GPU:0', 'memory_size': 8000000000, 'type': ''}, 
               1: {'name': '/job:localhost/replica:0/task:0/device:XLA_GPU:1', 'memory_size': 8000000000, 'type': ''}, 
               2: {'name': '/job:localhost/replica:0/task:0/device:XLA_GPU:2', 'memory_size': 8000000000, 'type': ''}, 
               3: {'name': '/job:localhost/replica:0/task:0/device:XLA_GPU:3', 'memory_size': 8000000000, 'type': ''}}


##########################################################################

"""
    we are going to use streams to allow parallel processing
"""
COMPUTE0 = torch.cuda.Stream(device=0)
COMPUTE1 = torch.cuda.Stream(device=1)
COMPUTE2 = torch.cuda.Stream(device=2)
COMPUTE3 = torch.cuda.Stream(device=3)
COMPUTE_STREAM = {0:COMPUTE0,1:COMPUTE1,2:COMPUTE2,3:COMPUTE3}


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------Utilities----------------------------------------------------------------------------------------

## Print memory of all available GPU's
def print_gpu_memory():
    for i in range(torch.cuda.device_count()):
        #print(torch.cuda.get_device_name(i))
        print("GPU:", i)
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(i)/1024**3,8), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(i)/1024**3,8), 'GB')
        #print("-----------------")
        #GPUtil.showUtilization()
        print("-----------")

# print memory of given GPU. ex: gpu_no = 0
def print_mem(gpu_id, cached=2, unit='GB'):
    if unit=='GB':
        mem_allocated = round(torch.cuda.memory_allocated(gpu_id)/1024**3,8)
        mem_cached    = round(torch.cuda.memory_reserved(gpu_id)/1024**3,8)
    else:
        mem_allocated = torch.cuda.memory_allocated(gpu_id)
        mem_cached    = torch.cuda.memory_reserved(gpu_id)
        
    if cached>0:
        print('Allocated:', mem_allocated , 'GB')
    if cached>1:
        print('Cached:   ', mem_cached    , 'GB')
    return mem_allocated, mem_cached

##########################################################################

# Get the leaf operations in a model. model.modules() gives not just the leaves, bbut higher levels as well
# Ref: https://stackoverflow.com/questions/54846905/pytorch-get-all-layers-of-model
# More explanation: https://discuss.pytorch.org/t/module-children-vs-module-modules/4551/4
def get_children(model: torch.nn.Module):
    # get children form model!
    children = list(model.children())
    flatt_children = {}
    if children == []:
        # if model has no children; model is last child! :O
        return {id(model): model}
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                flatt_children.update(get_children(child))
            except TypeError:
                flatt_children.update(get_children(child))
    return flatt_children

##########################################################################

def b2gb(x): return round(x/2**30,8)
class TorchTracemalloc():
    def __init__(self, gpu_id):
        self.gpu_id = gpu_id

    def __enter__(self):
        self.begin = torch.cuda.memory_allocated(self.gpu_id)
        torch.cuda.reset_max_memory_allocated(self.gpu_id) # reset the peak gauge to zero
        return self

    def __exit__(self, *exc):
        self.end  = torch.cuda.memory_allocated(self.gpu_id)
        self.peak = torch.cuda.max_memory_allocated(self.gpu_id)
        self.used   = (self.end-self.begin)
        self.peaked = (self.peak-self.begin)

#### Estimate size of the model (in GB or MB or )

def estimate_model_size(model, unit='MB', to_print = True): 
    persistent_memory = 0
    for name, param in model.named_parameters():
        persistent_memory += param.element_size() * param.nelement()
    if unit == 'GB':
        gb_mem = round(persistent_memory/1024**3,8)
        if to_print:
            print("Estimated Model Memory:",gb_mem, "GB")
        return gb_mem
    elif unit == 'B':
        gb_mem = persistent_memory
        if to_print:
            print("Estimated Model Memory:",gb_mem, "Bytes")
        return gb_mem
    else:
        mb_mem = round(persistent_memory/1024**2,8)
        if to_print:
            print("Estimated Model Memory:", mb_mem, "MB")
        return mb_mem
    
def estimate_tensor_size(inp, unit='B'):
    input_size = 0
    if isinstance(inp, torch.Tensor): 
        input_size += float(torch.prod(torch.tensor(inp.size())))
    if isinstance(inp, list): 
        for sub_inp in inp:
            if isinstance(sub_inp, torch.Tensor): input_size += float(torch.prod(torch.tensor(sub_inp.size())))

    input_size = input_size*torch.rand((1,1)).element_size() # multiply by 4
    if unit == 'GB':
        gb_mem = round(input_size/1024**3,8)
        #print("Estimated Input/Output Memory:",gb_mem, "GB")
        return gb_mem
    if unit == 'B':
        gb_mem = input_size
        #print("Estimated Input/Output Memory:",gb_mem, "B")
        return gb_mem
    else:
        mb_mem = round(input_size/1024**2,8)
        #print("Estimated Input/Output Memory:", mb_mem, "MB")
        return mb_mem

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------Baechi Units------------------------------------------------------------------------------------------------

class SubModuleNode:
    """
    This class represents a submodel (ex. conv2d layer) in the given model (ex. inception_v3). 
    It is represented as a node in the return graph
    """
    def __init__(self):
        # store the entire submodel
        self.module = None
        # submodel name
        self.name = None

        # nodes that must finish processing before this node (direct dependencies)
        self.parent = set()
        # nodes that depends on this node
        self.children = set()

        # forward function's estimated runtime
        self.weight_forward = 0
        # backward function's estimated runtime
        self.weight_backward = 0
        # id represented by the model's location (python's id function)
        self.id_hash = None
        # sudo id used, for one model, this sudo id starts from 0 and add 1 for each new node
        # -- self.id = None
        # storage used by submodel's parameters (weight, bias)
        self.persistent_memory = 0
        # submodel's input's size
        self.input_memory = 0
        # submodel's output's size
        self.output_memory = 0
        # temporary memory used in forward run
        self.temporary_memory = 0
        
        # gpu assigned to the submodule
        self.p = None
        
########################################################################

class Profiling:
    """
    This class produce the profile, this class referenced "https://github.com/msr-fiddle/pipedream"
    """
    def __init__(self, model, gpu=0, rounds=20, input_size=(3, 299, 299)):
        """
        model: ex. inception_v3 model, alexnet model, etc
        gpu: choose in between {0,1,2,3}
        rounds: number of rounds to run the profiling
        """
        self.gpu = gpu
        self.model = model.to(self.gpu)
        self.input_size = input_size

        self.rounds = rounds
        # first few rounds are inaccurate, so I choose to discard the results from the first 1/4 rounds
        self.ignore_rounds = int(self.rounds/4)
        # counting variable, runs from 0 - self.rounds
        self.cur_round = 0

        # used to calculate backward runtime for each submodule
        self.back_record = []
        # all submodules record of the form {id of the layer(submodule) : SubModuleNode created out of tha layer}
        self.sub_module_nodes = {}
        # use id_hash to record the order of submodules's execution
        self.submodule_order = []

        # internal use only, record the original forward functions for submodules
        self.forward_original_methods = {}
        # internal use only, switch back to the original forward functions after profiling
        self.detach_record = set()
        # Collect handles to all hooks added, so as to remove them in detach()
        self.hook_handles = []


    def recur_function(self, module):
        """
        modify self.model: adding forward timing, backward timing, input output sizes, etc
        :param module: the model to recursively add forward/backward wrappers to
        """
        this_profiler = self
        sub_modules = module.__dict__['_modules']
        for name, sub_module in sub_modules.items():
            # sub modules of sub_module, if there are more than 1, we need further recursion
            sub_sub_modules = sub_module.__dict__['_modules']
            if len(sub_sub_modules) > 0:
                self.recur_function(sub_module)
                continue
            
            def _calculate_time_and_memory(function, *input):
                """
                - Helper function in forward wrapper
                - Calculates forward runtime, peak memory used and static memory used
                - Verified: Memory measurement context doesn't add overhead to
                  time measurement
                """
                with TorchTracemalloc(self.gpu) as tt:
                    torch.cuda.synchronize(self.gpu)
                    start_time = time.time()
                    result = function(*input)
                    torch.cuda.synchronize(self.gpu)
                    stop_time = time.time()
                return (stop_time - start_time) * 1000, tt.used, tt.peaked , result

            def forward_wrapper(cur_module, *input):
                """
                use this wrapper to replace the original forward function in submodules
                :param cur_module: the input submodule
                """
                # original forward function
                
                function = this_profiler.forward_original_methods[cur_module]
                if this_profiler.cur_round < this_profiler.ignore_rounds:
                    if this_profiler.cur_round == 0:
                        # record submodule execution order only in the first round
                        # print('-->', "Module name: ",cur_module)
                        this_profiler.submodule_order.append(id(cur_module))
                    # do not record first few rounds
                    result = function(*input)
                    return result
                
                ## collect relevant information of cur module
                forward_time, used_mem, peak_mem, result = _calculate_time_and_memory(function, *input)
                
                ## Input size in bytes
                input_size = 0
                for inp in input:
                    input_size = input_size + estimate_tensor_size(inp, 'B')
                
                ## Model size in bytes
                persistent_memory = estimate_model_size(cur_module,'B', False)

                output_memory = estimate_tensor_size(result, 'B')
                
                '''
                if not(used_mem==512*np.ceil(output_memory/512)):
                    print('*'*50)
                    print("In sumodule ", cur_module , ':' )
                    print("Output memory is: ", output_memory)
                    print("But used memory is: ", used_mem)
                    print("They dont match upto a factor of 512 (since mem bolcks are alotted in 512 byte locks) as expected")
                    print('*'*50)
                '''
                    
                temporary_memory = peak_mem - used_mem

                # record a SubModuleNode for each model layer
                if id(cur_module) not in this_profiler.sub_module_nodes:
                    cur_node = SubModuleNode()
                    cur_node.id_hash = id(cur_module)
                    cur_node.module = cur_module
                    cur_node.name = cur_module.__class__.__name__
                    
                    #***********?????????????????????????????????????????***************************
                    ########## REMOVE THIS ######################
                    cur_node.persistent_memory = persistent_memory
                    cur_node.temporary_memory = temporary_memory
                    cur_node.output_memory = output_memory
                    cur_node.input_memory = input_size
                    #############################################
                    #***********?????????????????????????????????????????***************************
                    
                    ### And Uncomment this
                    #cur_node.persistent_memory = persistent_memory
                    #cur_node.temporary_memory = temporary_memory
                    #cur_node.output_memory = output_memory
                    #cur_node.input_memory = input_size
                    
                else:
                    cur_node = this_profiler.sub_module_nodes[id(cur_module)]
                # we want weight_forward as the average forward runtime of the relevent rounds
                cur_node.weight_forward += forward_time / (this_profiler.rounds - this_profiler.ignore_rounds)
                this_profiler.sub_module_nodes[id(cur_module)] = cur_node

                return result

            def hook(cur_module, inputs, output):
                # this is for retriving the module inside make dot function
                output.grad_fn.metadata['module'] = cur_module

            def backward_post_hook(cur_module, input, output):
                """
                add backward hook to record backward runtime
                :param cur_module: the input submodule
                """
                if this_profiler.cur_round < this_profiler.ignore_rounds:
                    # do not record first few rounds
                    return
                torch.cuda.synchronize(0)
                cur_time = time.time() * 1000
                this_profiler.back_record.append((id(cur_module), cur_time))

            if sub_module in self.forward_original_methods:
                # only record the original forward functions once
                continue

            self.forward_original_methods[sub_module] = sub_module.forward
            sub_module.forward = forward_wrapper.__get__(sub_module, sub_module.__class__)
            fhook_handle = sub_module.register_forward_hook(hook)
            bhook_handle =  sub_module.register_backward_hook(backward_post_hook)
            this_profiler.hook_handles.append(fhook_handle)
            this_profiler.hook_handles.append(bhook_handle)
            
            
    def detach(self, module):
        """
        use this helper function to detach all forward wrappers
        """
        this_profiler = self
        sub_modules = module.__dict__['_modules']
        for name, sub_module in sub_modules.items():
            sub_sub_modules = sub_module.__dict__['_modules']
            if len(sub_sub_modules) > 0:
                self.detach(sub_module)
                continue
            if sub_module in self.detach_record:
                continue

            self.detach_record.add(sub_module)
            sub_module.forward = self.forward_original_methods[sub_module]
        ## Remove all the hooks that were added
        for handle in this_profiler.hook_handles:
            handle.remove()

    def run(self):
        """
        :return: the model's output of the final round
        """
        self.sub_module_nodes = {}
        self.recur_function(self.model)

        # create a fake dataset, we don't care about accuracy.
        dataset = torchvision.datasets.FakeData(
            size=self.rounds * int(args.batch_size),
            #image_size=(3, 299, 299),
            image_size = self.input_size,
            num_classes=1000,
            transform=torchvision.transforms.ToTensor())
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=int(args.batch_size))

        # this is the output of the final round
        last_output = None
        for batch_idx, (inp, oup) in enumerate(data_loader):
            self.cur_round = batch_idx
            # clear the record list
            self.back_record = []

            inp = inp.to(self.gpu); inp.requires_grad = True
            optimizer = optim.SGD(self.model.parameters(), lr = 0.0001); optimizer.zero_grad()
            criterion = nn.MSELoss()
            
            torch.cuda.synchronize(self.gpu)
            output = self.model(inp)
            torch.cuda.synchronize(self.gpu)
             
            
            ######################### loss compute ################################################
            try:
                loss = criterion(output, torch.randn(int(args.batch_size), len(output)).to(self.gpu))
            except: ## For inception
                loss = criterion(output, torch.randn(int(args.batch_size), len(output[0])).to(self.gpu))
            ##################################################################################

            # add the start time of backward 
            self.back_record.append(('start', time.time() * 1000))
            if batch_idx == self.rounds - 1:
                #loss.backward(loss, retain_graph=True)
                loss.backward(loss)
                last_output = output
            else:
                loss.backward(loss)
            #print_mem(0,1)
            #print('*'*50)

            if batch_idx < self.ignore_rounds:
                continue
            else:
                # calculate the backward runtime for each layer by calculating the time differences between each timestamp
                for i in range(len(self.back_record) - 1, 0, -1):
                    now = self.back_record[i]
                    prev = self.back_record[i - 1]
                    cur_node = self.sub_module_nodes[now[0]]
                    cur_node.weight_backward += (now[1] - prev[1]) / (self.rounds - self.ignore_rounds)
                    self.sub_module_nodes[now[0]] = cur_node
        self.detach(self.model)
        return last_output

########################################################################

'''
make_dot is modified to add nodes for only the autograd corresponding to layers
'''

def make_dot(var, cur_model):
    """
    this function build a DiGraph for the model, by tracing the grad function of each layer's output
    :return: the DiGraph
    """
    dot = nx.DiGraph()
    seen = set()
    output_nodes = (var.grad_fn,) if not isinstance(var, tuple) else tuple(v.grad_fn for v in var)

    def add_nodes(var):
        # print("Dealing with this variable:", var)
        if var not in seen:
            cur_id = None
            if var.metadata != {}:
                if ('module' in var.metadata):
                    # this submodule has a forward function, so it's information is previously recorded in Profiling
                    cur_id = id(var.metadata['module'])
                    # retrieve the node representing this submodule
                    cur_node = cur_model.sub_module_nodes[id(var.metadata['module'])]
                    dot.add_node(id(var.metadata['module']), 
                                 model = str(cur_node.module), 
                                 name = str(cur_node.name), 
                                 weight=cur_node.weight_forward,
                                 reverse_weight=cur_node.weight_backward,
                                 id=id(var.metadata['module']), 
                                 topo_order=id(var.metadata['module']), 
                                 temporary_memory=cur_node.temporary_memory, 
                                 persistent_memory=cur_node.persistent_memory,
                                 output_memory=[cur_node.output_memory], 
                                 output_tensors=cur_node.output_memory, 
                                 colocation_group="")
                    
                    if hasattr(var, 'next_functions'):
                        for u in var.next_functions:
                            if u[0] is not None and torch.is_tensor(u[0]) is False and hasattr(u[0], 'variable') is False:
                                if u[0].metadata != {}:
                                    if ('module' in u[0].metadata):
                                        next_id = id(u[0].metadata['module'])
                                        cur_model.sub_module_nodes[next_id].children.add(cur_id)
                                        cur_model.sub_module_nodes[cur_id].parent.add(next_id)
                                    elif ('parent' in u[0].metadata):
                                        u[0].metadata['parent'].add(cur_id)
                                    else:
                                        print("Error:", u[0], " has metadata that is neither module nor parent!")
                                        return 0
                                else:
                                    u[0].metadata['parent'] = set()
                                    u[0].metadata['parent'].add(cur_id)
                                    
                                add_nodes(u[0])
                                
                elif ('parent' in var.metadata):
                    cur_id_list = []
                    for parent in var.metadata['parent']:
                        cur_id_list.append(parent)
                    if hasattr(var, 'next_functions'):
                        for u in var.next_functions:
                            if u[0] is not None and torch.is_tensor(u[0]) is False and hasattr(u[0], 'variable') is False:
                                if u[0].metadata != {}:
                                    if ('module' in u[0].metadata):
                                        next_id = id(u[0].metadata['module'])
                                        for cur_id in cur_id_list:
                                            cur_model.sub_module_nodes[next_id].children.add(cur_id)
                                            cur_model.sub_module_nodes[cur_id].parent.add(next_id)
                                    elif ('parent' in u[0].metadata):
                                        for cur_id in cur_id_list:
                                            u[0].metadata['parent'].add(cur_id)
                                    else:
                                        print("Error:", u[0], " has metadata that is neither module nor parent!")
                                        return 0
                                else:
                                    u[0].metadata['parent'] = set()
                                    for cur_id in cur_id_list:
                                        u[0].metadata['parent'].add(cur_id)
                                add_nodes(u[0])
                
            else:
                ## All functions will have either 'module' or 'parent' metadata
                print('*'*100)
                print("Error:", var, " does not have any metadata!")
                print(var.__dict__)
                print('*'*100)
                return 0

            seen.add(var)

    if isinstance(var, tuple):
        # handle multiple outputs
        for v in var:
            add_nodes(v.grad_fn)
    else:
        add_nodes(var.grad_fn)
    
    return dot

#----------------------------------------------------------Baechi Utilities-----------------------------------------------------------------------------------------------------

def topological_sort(model):
    """
    this helper function helps to generate the execution order based on dependecies
    """
    record = set()
    while len(record) < len(model.sub_module_nodes):
        #print(len(record),len(model.sub_module_nodes) )
        root_helper = set(model.sub_module_nodes.keys()) - record
        reordered_root_helper = []
        for elem in model.submodule_order:
            if elem in root_helper:
                reordered_root_helper.append(elem)
        reordered_root_helper += list(root_helper - set(reordered_root_helper))
        for elem in root_helper:
            parents = model.sub_module_nodes[elem].parent
            if parents is None or len(parents - record) == 0:
                model.sub_module_nodes[elem].id = len(record)
                record.add(elem)


def copy_p(assigned_graph, model):
    """
    helper function to add .p field based on the assigned DiGraph 
    """
    for node_id in model.sub_module_nodes:
        model.sub_module_nodes[node_id].p = assigned_graph.nodes[model.sub_module_nodes[node_id].id]["p"]

        
def recursively_assign(Input, Device):
    """
    helper function to assign Input recursively to a gpu Device
    """
    result = None
    if isinstance(Input, list):
        result = []
        for elem in Input:
            result.append(recursively_assign(elem, Device))
    else:
        if Input.device.index != Device:
            result = Input.cuda(Device)
        else:
            result = Input
    return result


def print_assigned_graph(return_graph):
    """
    helper function to print where each layer is assigned to
    :param return_graph: assigned DiGraph
    """
    my_record = {}
    for node in return_graph.nodes(data=True):
        my_record[node[1]['topo_order']] = (node[1]['name'], node[1]['persistent_memory'], node[1]['weight'], node[1]['p'])

    for i in range(len(return_graph.nodes)):
        print(i, my_record[i])

#----------------------------------------------------Baechi Main Units--------------------------------------------------------------------------------------------------------------

def build_graph(model, gpu=0, rounds=1, inp_size = (3,299,299)):
    """
    this is the main function to call for building the graph, it calls profiling and make dot, and made further improvements
    :param model: input model (ex. inception_v3)
    :param gpu: which gpu to place the profiler
    :param rounds: number of rounds to run the profiling
    :return: the DiGraph, and the Profiling object
    """
    #print("Profiling started", '*'*20)
    tester = Profiling(model, gpu, rounds, input_size = inp_size)
    final_output = tester.run()
    
    #print("make_dot started", '*'*20)
    return_graph = make_dot(final_output, tester)

    #print("Sort topologically", '*'*20)
    topological_sort(tester)
    
    # use the sudo id instead of hash_id, this is for scheduler purpose
    #print("Replacing sub module id", '*'*20)
    for node_id in tester.sub_module_nodes.keys():
        model_node = tester.sub_module_nodes[node_id]
        #if len(model_node.parent)==0 and len(model_node.children)==0:
        #    print('*'*50)
        #    print("Module defined but not used:")
        #    print(model_node.__dict__)
        #    print('*'*50)
        #else:
        graph_node = return_graph.nodes[node_id]
        graph_node["id"] = model_node.id
        graph_node["topo_order"] = model_node.id
        return_graph.add_nodes_from([(model_node.id, graph_node)])
        return_graph.remove_node(node_id)

    # since some 'sub module' have no forward function, we have limited data on them. So we assume their output's size to be their parent's output's size
    for new_id in range(len(return_graph.nodes(data=True))):
        for node_id in tester.sub_module_nodes.keys():
            if tester.sub_module_nodes[node_id].id == new_id:
                old_id = node_id
        for child_old_id in tester.sub_module_nodes[old_id].children:
            child_new_id = tester.sub_module_nodes[child_old_id].id
            if return_graph.nodes[child_new_id]['weight'] < 0.001:
                tester.sub_module_nodes[child_old_id].output_memory = return_graph.nodes[new_id]['output_tensors']
                tester.sub_module_nodes[child_old_id].input_memory = return_graph.nodes[new_id]['output_tensors']
                return_graph.nodes[child_new_id]['output_tensors'] = return_graph.nodes[new_id]['output_tensors']
                return_graph.nodes[child_new_id]['output_memory'] = return_graph.nodes[new_id]['output_memory'].copy()

    # change the id of edges
    #print("Filling in the edges", '*'*20)
    edge_count = 0
    for node in tester.sub_module_nodes.keys():
        children = tester.sub_module_nodes[node].children
        node_new_id = tester.sub_module_nodes[node].id
        for kid in children:
            kid_new_id = tester.sub_module_nodes[kid].id
            edge_data = {
                "weight": 0, "id": edge_count, "tensor": [],
                # the requested bytes <= min(from_node's output size, to_node's input size)
                "requested_bytes": min(tester.sub_module_nodes[node].output_memory, tester.sub_module_nodes[kid].input_memory),
            }
            if edge_data['requested_bytes'] != 0:
                # set the weight of the data base on 7.3 * (10 ^ (-7)) * x 
                edge_data['weight'] = 7.3 * (edge_data['requested_bytes'] / (10 ** 7))
            edge_data['tensor'] = [{"name": str(edge_count), "recv_end_ts": 0, "weight": edge_data['weight']}]
            return_graph.add_edge(node_new_id, kid_new_id, **edge_data)
            #return_graph.add_edge(kid_new_id, node_new_id, **edge_data)
            edge_count += 1
            
    if args.type == "all":
        print("Yet to implement")
        return 0, 0

    return return_graph, tester

class Assign(object):
    """
    This class actually put each submodule to the gpu it is assigned to
    """
    def __init__(self, model_wrapper):
        self.model = model_wrapper
        self.original_forwards = {}
        self.assigned = self.recur_move_layers_to_gpus(model_wrapper.model)
    
    def recur_move_layers_to_gpus(self, module):
        
        this_assigner = self
        sub_modules = module.__dict__['_modules']
        if len(sub_modules) > 0:
            for name, sub_module in sub_modules.items():
                this_assigner.recur_move_layers_to_gpus(sub_module)
        else:
            module_id = id(module)
            gpu_id = this_assigner.model.sub_module_nodes[module_id].p
    
            ### Move layers to the allotted GPUs
            # module.to(gpu_id)
            ########### FOR TESTING ##################################################
            mem0, _ = print_mem(gpu_id, cached=0, unit='B')
            module.to(gpu_id)
            mem1, _ = print_mem(gpu_id, cached=0, unit='B')
            # print("Module:              ", module)
            # print("GPU:                 ", gpu_id)
            # print("Memory change:       ", mem1-mem0)
            # print("Layer size:          ", estimate_model_size(module, unit='B', to_print=False))
            # print("Net memory occupied: ", mem1)
            # print("*"*50)
            #########################################################################
            
            this_assigner.original_forwards[module_id] = module.forward

            def modified_forward(self, *inputs):
                #print(self)
                #print(gpu_id)
                #print('*'*50)
                #start = time.time()
                #print(this_assigner.model.sub_module_nodes[module_id].parent)

                #########################################################
                input_list = list(inputs)
                for i, inp in enumerate(input_list):
                    if isinstance(inp, torch.Tensor):
                        #print("Getting input from ", inp.get_device(), " to ", gpu_id)
                        if 1:          
                        #with torch.cuda.stream(torch.cuda.Stream(device=inp.get_device())):
                        #with torch.cuda.stream(COMPUTE_STREAM[inp.get_device()]):  ## Stream must be setup on the sending node (see experiment.ipynb, section "Use of CUDA Streams")
                            #with torch.cuda.stream(COMPUTE_STREAM[gpu_id]):  ### ??? IS THIS SAFE ???
                            #with torch.cuda.stream(torch.cuda.Stream(device=gpu_id)):
                            if 1:
                                input_list[i] = inp.to(gpu_id)
                    else:
                        pass
                #         print("Input not a Tensor!") ## Fix this
                inputs = tuple(input_list)
                ########################################################
                #print("Input transfer time: ",  time.time() - start)
                output = this_assigner.original_forwards[module_id](*inputs) 
                #print("Net time: ",  time.time() - start)
                #print('*'*50)
                # id(self) = module_id since forward is method of a module
                return output

            module.forward =  modified_forward.__get__(module, module.__class__)  