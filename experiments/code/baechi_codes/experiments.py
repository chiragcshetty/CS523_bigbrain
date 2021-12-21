"""
    A sample command line: python3 summarize.py --prof_rounds 4 --prof_gpu_id 0 --gpu_num 1 --sch sct --batch_size 32 --type all
    Read the last two lines of output to get (1) virtual runtime (2) real runtime
"""
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
sys.path.append('/home/cshetty2/sct/pytorch')
import reformated_models.pytorch_modified_inception as pytorch_modified_inception
import simple_model as sm
sys.path.append('/home/cshetty2/sct')
from placer.placer_lib import *

"""
    type: take into consideration both forward and backward, or merely forward
    prof_rounds: number of rounds to run the profiling, increase this number to increase the accuracy, but also take into consideration the storage of gpu.
    prof_gpu_id: which gpu to place the profiler, choose in between {0,1,2,3}, this shouldn't have huge impact on your result
    batch_size: the batch size of the model
    gpu_num: number of gpus the model will be distributed on, choose in between {1,4}
    sch: the scheduler wanted, choose in between {sct,etf,topo}
"""
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--type', type=str, help='forward/all')
parser.add_argument('--prof_rounds', type=int, help='rounds for profiler')
parser.add_argument('--prof_gpu_id', type=int, help='which gpu to place the profiler')
parser.add_argument('--batch_size', type=str, help='batch_size')
parser.add_argument('--gpu_num', type=int, help='number of gpu to use')
parser.add_argument('--sch', type=str, help='sct/etf/topo')

"""
    device graphs for single gpus case and 4 gpu case
"""
DEVICE_GRAPH_SINGLE = create_device_graph({0: {'name': '/job:localhost/replica:0/task:0/device:XLA_GPU:0', 'memory_size': 17179869184, 'type': ''}})
DEVICE_GRAPH_MULTIPLE = create_device_graph({0: {'name': '/job:localhost/replica:0/task:0/device:XLA_GPU:0', 'memory_size': 8000000000, 'type': ''}, 
                                             1: {'name': '/job:localhost/replica:0/task:0/device:XLA_GPU:1', 'memory_size': 8000000000, 'type': ''}, 
                                             2: {'name': '/job:localhost/replica:0/task:0/device:XLA_GPU:2', 'memory_size': 8000000000, 'type': ''}, 
                                             3: {'name': '/job:localhost/replica:0/task:0/device:XLA_GPU:3', 'memory_size': 8000000000, 'type': ''}})

"""
    we are going to use streams to allow parallel processing
"""
COMPUTE0 = torch.cuda.Stream(device=0)
COMPUTE1 = torch.cuda.Stream(device=1)
COMPUTE2 = torch.cuda.Stream(device=2)
COMPUTE3 = torch.cuda.Stream(device=3)
COMPUTE_STREAM = {0:COMPUTE0,1:COMPUTE1,2:COMPUTE2,3:COMPUTE3}

class SubModuleNode:
    """
    This class represents a submodel (ex. conv2d layer) in the given model (ex. inception_v3). It is represented as a node in the return graph
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
        self.id = None
        # storage used by submodel's parameters (weight, bias)
        self.persistent_memory = 0
        # submodel's input's size
        self.input_memory = 0
        # submodel's output's size
        self.output_memory = 0
        
        # gpu assigned to
        self.p = None
       
                
class Profiling:
    """
    This class produce the profile, this class referenced "https://github.com/msr-fiddle/pipedream"
    """
    def __init__(self, model, gpu=0, rounds=20, inception=False):
        """
        model: ex. inception_v3 model, alexnet model, etc
        gpu: choose in between {0,1,2,3}
        rounds: number of rounds to run the profiling
        """
        self.gpu = gpu
        self.model = model.to(self.gpu)

        self.rounds = rounds
        # first few rounds are inaccurate, so I choose to discard the results from the first 1/4 rounds
        self.ignore_rounds = int(self.rounds/4)
        # counting variable, runs from 0 - self.rounds
        self.cur_round = 0

        # used to calculate backward runtime for each submodule
        self.back_record = []
        # all submodules record of the form {id_hash: SubModuleNode}
        self.sub_module_nodes = {}
        # use id_hash to record the order of submodules's execution
        self.submodule_order = []

        # internal use only, record the original forward functions for submodules
        self.forward_original_methods = {}
        # internal use only, switch back to the original forward functions after profiling
        self.detach_record = set()
        # Indicates if it is an inception model
        self.inception = inception

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

            def calculate_forward_time(function, *input):
                """
                helper function in forward wrapper
                calculate forward runtime, and submodule result
                """
                torch.cuda.synchronize(self.gpu)
                start_time = time.time()
                result = function(*input)
                torch.cuda.synchronize(self.gpu)
                stop_time = time.time()
                return (stop_time - start_time) * 1000, result

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
                        this_profiler.submodule_order.append(id(cur_module))
                    # do not record first few rounds
                    result = function(*input)
                    return result
                
                # collect relevant information of cur module
                forward_time, result = calculate_forward_time(function, *input)
                input_size = 0; persistent_memory = 0; output_memory = 1
                for inp in input:
                    if isinstance(inp, torch.Tensor): input_size += float(torch.prod(torch.tensor(inp.size())))
                    if isinstance(inp, list): 
                        for sub_inp in inp:
                            if isinstance(sub_inp, torch.Tensor): input_size += float(torch.prod(torch.tensor(sub_inp.size())))
                for name, param in cur_module.named_parameters():
                    product = 1
                    for i in param.size(): product *= i
                    persistent_memory += product
                for i in result.size(): output_memory *= i

                # record a SubModuleNode for each model layer
                if id(cur_module) not in this_profiler.sub_module_nodes:
                    cur_node = SubModuleNode()
                    cur_node.id_hash = id(cur_module)
                    cur_node.module = cur_module
                    cur_node.name = cur_module.__class__.__name__
                    cur_node.persistent_memory = persistent_memory
                    cur_node.output_memory = output_memory
                    cur_node.input_memory = input_size
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
            sub_module.register_forward_hook(hook)
            sub_module.register_backward_hook(backward_post_hook)
            
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

    def run(self):
        """
        :return: the model's output of the final round
        """
        self.sub_module_nodes = {}
        self.recur_function(self.model)

        # create a fake dataset, we don't care about accuracy.
        dataset = torchvision.datasets.FakeData(
            size=self.rounds * int(args.batch_size),
            image_size=(3, 299, 299),
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

            ## the following two lines only apply to inception_v3 model
            if (self.inception):
                output = output.logits[0] + output.aux_logits[0]
                loss = criterion(output, torch.randn(len(output)).to(self.gpu))
            else: ## if the model is not inception_v3, use the following one line
                loss = criterion(output, torch.randn(int(args.batch_size), len(output[0])).to(self.gpu))
            
            
            
            # add the start time of backward 
            self.back_record.append(('start', time.time() * 1000))
            if batch_idx == self.rounds - 1:
                loss.backward(loss, retain_graph=True)
                last_output = output
            else:
                loss.backward(loss)

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
    
    
def topological_sort(model):
    """
    this helper function helps to generate the execution order based on dependecies
    """
    record = set()
    while len(record) < len(model.sub_module_nodes):
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
                
    
def make_dot(var, cur_model):
    """
    this function build a DiGraph for the model, by tracing the grad function of each layer's output
    :return: the DiGraph
    """
    dot = nx.DiGraph()
    seen = set()
    output_nodes = (var.grad_fn,) if not isinstance(var, tuple) else tuple(v.grad_fn for v in var)

    def add_nodes(var):
        if var not in seen:
            cur_id = None
            if var.metadata != {}:
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
                             temporary_memory=0, 
                             persistent_memory=cur_node.persistent_memory,
                             output_memory=[cur_node.output_memory], 
                             output_tensors=cur_node.output_memory, 
                             colocation_group="")
            else:
                # this 'submodule' has no forward function, we assume that the forward runtime & backward runtime can be ignored
                cur_id = id(var)
                dot.add_node(id(var), 
                             model = None, 
                             name = str(type(var).__name__),
                             weight=0.000000001, 
                             reverse_weight=0.000000001,
                             id=id(var), 
                             topo_order=id(var),
                             temporary_memory=0, 
                             persistent_memory=0, 
                             output_memory=[0], 
                             output_tensors=0, 
                             colocation_group="")
                if cur_id not in cur_model.sub_module_nodes:
                    # should only occur once for the final output
                    represent_node = SubModuleNode()
                    represent_node.name = str(type(var).__name__)
                    represent_node.id_hash = cur_id
                    cur_model.sub_module_nodes[cur_id] = represent_node
            seen.add(var)

            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None and torch.is_tensor(u[0]) is False and hasattr(u[0], 'variable') is False:
                        next_id = id(u[0])
                        if u[0].metadata != {}: 
                            next_id = id(u[0].metadata['module'])
                        else:
                            # append a new node to model's record if not seen before
                            represent_node = SubModuleNode()
                            represent_node.name = str(type(u[0]).__name__)
                            represent_node.id_hash = id(u[0])
                            cur_model.sub_module_nodes[id(u[0])] = represent_node
                        cur_model.sub_module_nodes[next_id].children.add(cur_id)
                        cur_model.sub_module_nodes[cur_id].parent.add(next_id)
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    if torch.is_tensor(t) or hasattr(t, 'variable'): continue
                    next_id = id(t)
                    if t.metadata != {}:
                        next_id = id(t.metadata['module'])
                    else:
                        # append a new node to model's record if not seen before
                        represent_node = SubModuleNode()
                        represent_node.name = str(type(t).__name__)
                        represent_node.id_hash = id(t)
                        cur_model.sub_module_nodes[id(t)] = represent_node
                    cur_model.sub_module_nodes[next_id].children.add(cur_id)
                    cur_model.sub_module_nodes[cur_id].parent.add(next_id)
                    add_nodes(t)

    if isinstance(var, tuple):
        # handle multiple outputs
        for v in var:
            add_nodes(v.grad_fn)
    else:
        add_nodes(var.grad_fn)
    
    return dot

def build_graph(model, gpu=0, rounds=1, inception=False):
    """
    this is the main function to call for building the graph, it calls profiling and make dot, and made further improvements
    :param model: input model (ex. inception_v3)
    :param gpu: which gpu to place the profiler
    :param rounds: number of rounds to run the profiling
    :return: the DiGraph, and the Profiling object
    """
    tester = Profiling(model, gpu, rounds, inception)
    final_output = tester.run()
    return_graph = make_dot(final_output, tester)

    topological_sort(tester)
    
    # use the sudo id instead of hash_id, this is for scheduler purpose
    for node_id in tester.sub_module_nodes.keys():
        model_node = tester.sub_module_nodes[node_id]
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
            edge_count += 1
            
    if args.type == "all":
        # if both forward and backward operations are required by user, we have to also add in backward nodes. Which is done through basically mirroring the existing nodes and edges
        modify_list_nodes = list(return_graph.nodes(data=True)).copy()
        modify_list_edges = list(return_graph.edges(data=True)).copy()
        for Node in modify_list_nodes:
            node = Node[1]
            backward_id = 2*len(modify_list_nodes)-node["id"]-1
            return_graph.add_node(backward_id,
                                  name = "B_" + node["name"],
                                  weight=node["reverse_weight"],
                                  reverse_weight=node["weight"],
                                  id=backward_id,
                                  topo_order=backward_id,
                                  temporary_memory=0,
                                  persistent_memory=0,
                                  output_memory=[node["persistent_memory"]],
                                  output_tensors=node["persistent_memory"],
                                  colocation_group=str(node["id"]))

        for edge in modify_list_edges:
            new_in_edge = 2*len(modify_list_nodes)-edge[1]-1
            new_out_edge = 2*len(modify_list_nodes)-edge[0]-1 
            edge_data = edge[2].copy()
            edge_data['id'] = edge_count
            edge_data['requested_bytes'] = return_graph.nodes[new_in_edge]["output_tensors"]
            if edge_data['requested_bytes'] != 0:
                edge_data['weight'] = 3.578937350225824e-07 * edge_data['requested_bytes'] + 0.11357127074681594
            edge_data['tensor'] = [{"name":str(edge_count), "recv_end_ts":0, "weight":edge_data['weight']}]
            edge_count += 1
            return_graph.add_edge(new_in_edge, new_out_edge, **edge_data)

    return return_graph, tester


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


class Assign(object):
    """
    This class actually put each submodule to the gpu it is assigned to
    """
    def __init__(self, model_wrapper):
        self.model = model_wrapper
        self.forward_original_methods = {}
        # record the output from each layer. This is used so that it is possible to transfer data before it is required by another submodule
        self.record = {}
        # to identify if a submodule is in its first round
        self.first_round = set()
        self.assigned = self.recur_function(model_wrapper.model)

    def recur_function(self, module):
        """
        recursively assign the model to the given gpu
        """
        this_profiler = self
        sub_modules = module.__dict__['_modules']
        for name, sub_module in sub_modules.items():
            sub_module_name = sub_module.__class__.__name__
            sub_sub_modules = sub_module.__dict__['_modules']
            if len(sub_sub_modules) > 0:
                self.recur_function(sub_module)
                continue

            def forward_wrapper(self, *input):
                node_wrapper = this_profiler.model.sub_module_nodes[id(self)]
                node_gpu = node_wrapper.p
                
                with torch.cuda.stream(COMPUTE_STREAM[node_gpu]):
                    input = list(input)
                    for i in range(len(input)):
                        inp = input[i]
                        if isinstance(inp, torch.Tensor):
                            if id(inp) in this_profiler.record and node_gpu in this_profiler.record[id(inp)]:
                                # already transfered to the gpu we needed
                                if tuple(inp.size()) == tuple(this_profiler.record[id(inp)][node_gpu].size()):
                                    # could be used since there were no other modification
                                    input[i] = this_profiler.record[id(inp)][node_gpu]      
                                else:
                                    input[i] = recursively_assign(inp, node_gpu)
                            else:
                                input[i] = recursively_assign(inp, node_gpu)
                        elif isinstance(inp, list):
                            for j, sub_inp in enumerate(inp):
                                if id(sub_inp) in this_profiler.record and node_gpu in this_profiler.record[id(sub_inp)]:
                                    # already transfered to the gpu we needed
                                    if tuple(sub_inp.size()) == tuple(this_profiler.record[id(sub_inp)][node_gpu].size()):
                                        # could be used since there were no other modification
                                        input[i][j] = this_profiler.record[id(sub_inp)][node_gpu]      
                                    else:
                                        input[i][j] = recursively_assign(sub_inp, node_gpu)
                                else:
                                    input[i][j] = recursively_assign(sub_inp, node_gpu)
                    input = tuple(input)

                function = this_profiler.forward_original_methods[self]
                if id(self) not in this_profiler.first_round:
                    this_profiler.first_round.add(id(self))
                    with torch.cuda.stream(COMPUTE_STREAM[node_gpu]):
                        # only transfer the whole model to the gpu once, deal with parameter transfers only for all other rounds
                        self.cuda(node_gpu)
                    
                with torch.cuda.stream(COMPUTE_STREAM[node_gpu]):    
                    # put the computation on different streams to allow parallel computation
                    result = function(*input)
                
                this_profiler.record[id(result)] = {}
                this_profiler.record[id(result)][node_gpu] = result
                for kid_id in node_wrapper.children:
                    kid_node_wrapper = this_profiler.model.sub_module_nodes[kid_id]
                    if kid_node_wrapper.p not in this_profiler.record[id(result)]:
                        with torch.cuda.stream(COMPUTE_STREAM[kid_node_wrapper.p]):
                            # start trasnfering output to gpus needed as soon as output is produced
                            this_profiler.record[id(result)][kid_node_wrapper.p] = result.to(kid_node_wrapper.p)
                
                return result

            if sub_module in self.forward_original_methods:
                continue

            self.forward_original_methods[sub_module] = sub_module.forward
            sub_module.forward = forward_wrapper.__get__(sub_module, sub_module.__class__)

def print_gpu_memory():
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(i)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(i)/1024**3,1), 'GB')
        print("-----------------")
            
            
if __name__ == "__main__":
    args = parser.parse_args()

    inception = False
    
    if inception:
        dl_model = pytorch_modified_inception.inception_v3(pretrained=True)
    else:
        dl_model = sm.simpleModel(factor=1)

    return_graph, tester = build_graph(dl_model, args.prof_gpu_id, args.prof_rounds, inception)
    device_choice = args.gpu_num

    if device_choice == 1:
        if args.sch == "sct":
            placed_op_graph = m_sct(return_graph, DEVICE_GRAPH_SINGLE)
        elif args.sch == "etf":
            placed_op_graph = m_etf(return_graph, DEVICE_GRAPH_SINGLE)
        else:
            placed_op_graph = m_topo(return_graph, DEVICE_GRAPH_SINGLE)
        scheduler = VirtualScheduler(placed_op_graph, DEVICE_GRAPH_SINGLE)
    else:
        if args.sch == "sct":
            placed_op_graph = m_sct(return_graph, DEVICE_GRAPH_MULTIPLE)
        elif args.sch == "etf":
            placed_op_graph = m_etf(return_graph, DEVICE_GRAPH_MULTIPLE)
        else:
            placed_op_graph = m_topo(return_graph, DEVICE_GRAPH_MULTIPLE)
        scheduler = VirtualScheduler(placed_op_graph, DEVICE_GRAPH_MULTIPLE, True)
    copy_p(return_graph, tester)
    scheduler.initialize()
    result = scheduler.run()
    # second last print line
    print("virtual scheduler result: {}".format(result))

    #####*************************************************************************
    print("Allotment")
    allotment ={}
    for gpu_idx in range(torch.cuda.device_count()):
        allotment[gpu_idx] = []
    for node_id in tester.sub_module_nodes:
        allotment[tester.sub_module_nodes[node_id].p].append(tester.sub_module_nodes[node_id].module)
    for gpu_idx in range(torch.cuda.device_count()):
        print("GPU:", gpu_idx)
        print(allotment[gpu_idx], "\n\n")

    #####*************************************************************************
    
    Assign(tester)
    optimizer = optim.SGD(tester.model.parameters(), lr = 0.0001); optimizer.zero_grad()
    criterion = nn.MSELoss()
    dataset = torchvision.datasets.FakeData(
        size= args.prof_rounds * int(args.batch_size),
        image_size=(3, 299, 299),
        num_classes=1000,
        transform=torchvision.transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=int(args.batch_size))
    result = []
    for batch_idx, (inp, oup) in enumerate(data_loader):
        if(1):
            print("Batch: ", batch_idx)
            #print_gpu_memory()
            
            # try and except are needed to avoid gpu out of memory
            inp = inp.to(0); inp.requires_grad = True
            torch.cuda.synchronize(0);torch.cuda.synchronize(1);torch.cuda.synchronize(2);torch.cuda.synchronize(3)
            start_time = time.time()
            output = tester.model(inp)
            torch.cuda.synchronize(0);torch.cuda.synchronize(1);torch.cuda.synchronize(2);torch.cuda.synchronize(3)
            end_time = time.time()
            result_time = end_time - start_time
            if args.type == "all":
                if inception:
                    output = output.logits[0] + output.aux_logits[0].to(output.logits[0].device)
                loss = criterion(output, torch.randn(len(output)).to(output.device))
                torch.cuda.synchronize(0);torch.cuda.synchronize(1);torch.cuda.synchronize(2);torch.cuda.synchronize(3)
                start_time = time.time()
                loss.backward(loss)
                torch.cuda.synchronize(0);torch.cuda.synchronize(1);torch.cuda.synchronize(2);torch.cuda.synchronize(3)
                end_time = time.time()
                result_time += end_time - start_time
            
            #####*************************************************************************    
            GPUtil.showUtilization()
            del loss 
            del output
            torch.cuda.empty_cache()
            print("_______________")
            GPUtil.showUtilization()
            print("-********************************************")
            #####*************************************************************************

            result.append((result_time) * 1000)
        #except:
        #    print ("Something Broke!")
        #    break
    # last print line
    try:
        result_that_counts = result[1:]
        print("real runtime result: {}, with {} test runs".format(sum(result_that_counts)/len(result_that_counts), len(result_that_counts)))
    except:
        print("sorry, failed to reach at least 2 test runs, here are your results: {}".format(result))