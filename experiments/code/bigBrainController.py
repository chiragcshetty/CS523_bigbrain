import torch
import torchvision
import time
from torch import optim, nn
import numpy as np

import sys
import dummyModels as dm
import gc

import threading
from random import expovariate

#import matplotlib.pyplot as plt
import argparse


######## For profiler (some experiments. Not required) #################
from torch.profiler import profile, record_function, ProfilerActivity

###############################Utilities################################
        
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

print_gpu_memory()
####################### Parse Args ##################################
parser = argparse.ArgumentParser()

parser.add_argument("--rate", help="Job arrival rate", type=float, default=0.5)
parser.add_argument("--Texp", help="How long to run the exp in sec", type=int, default=100)
parser.add_argument("--algo", help="Algoritm for Packing", type=str, default='baseline')
parser.add_argument("--factor", help="Proxy for model size", type=int, default=6)
parser.add_argument("--batch_size", help="Batch size", type=int, default=32)
parser.add_argument("--samples", help="Number of batches", type=int, default=1000)

args = parser.parse_args()

rate: float = args.rate
fct: int = args.factor
batch_size: int = args.batch_size
Nrun: int = args.samples
algo: str = args.algo
T_exp: int = args.Texp

# ### Defaults ###
# T_exp      = 100         # (in sec) Time for which the exp will run
# rate       = 0.5         # job arrival rate
# batch_size = 32          # Size of mini batch
# Nrun       = 30          # number of mini-batches in one run
# fct        = 6           # factor, used to scale the model used
# algo       = 'baseline'  # baseline/bigbrain

####################### GLOBALS #########################################

### dict of (gpu_id, fraction of gpu available). Assumes 2 GPUs
resource_manager   = {}  
resource_manager[0]= 1
resource_manager[1]= 1

### dict of job_id -> dict{model_split, status_flag, arrival_time, entry_time, exit_time}
job_queue          = {}  

end_exp_flag = [0] #flag to indicate end of experiment
###################### Algorithms ########################################

# A job gets 1 GPU
def get_split_algo_1():
    if resource_manager[0] == 1:
        return [0,0]
    elif resource_manager[1] == 1:
        return [1,1]
    else:
        return None   

# Jobb gets one GPU (i.e [1,1],[0,0]) or maybe split across gpus (i.e [0,1])
# Assumes one whole job takes 0.7 of a gpu. Splitting takes 0.3 gpu on each gpu    
def get_split_algo_2():
    if resource_manager[0] > 0.5:
        return [0,0]
    elif resource_manager[1] > 0.5:
        return [1,1]
    elif (resource_manager[0] > 0.2 and resource_manager[1] > 0.2):
        return [0,1]
    return None
############## RESOURCE MANAGER ##############################################

def update_resource(split, update_type):
    if update_type == "release":
        change =  1.0
    elif update_type == "acquire":
        change = -1.0
     
    if split == [0,0]:
        resource_manager[0] = resource_manager[0] + 0.7*change
    elif split == [1,1]:
        resource_manager[1] = resource_manager[1] + 0.7*change
    elif split == [0,1]:
        resource_manager[0] = resource_manager[0] + 0.3*change
        resource_manager[1] = resource_manager[1] + 0.3*change
    else:
        print("Error! split is invalid")

############### Job Arrival: Poisson #######################################
## When a job arrives, create a new key in job_queue dict
## Entry is a job dict with attributes:
## model_split  = gpu allotment for the job
## status_flag  = [0] means yet to complete. Complete when [non_zero]
## arrival_time = [time when job was created]
## entry_time   = [time when job leaves the queue and starts executing]
## exit_time    = [time when job completes]
def job_arrivals(rate, T_exp):
    exp_start_time  = time.time()
    t = exp_start_time 
    job_id = 0
    
    while t < exp_start_time + T_exp:
        time.sleep(expovariate(rate))
        job_id = job_id+1
        job_queue[job_id] = {'model_split':None, 'status_flag':[0], \
                             'arrival_time':[time.time()], 'entry_time':[0], 'exit_time':[0]}
        t = time.time()
        
    end_exp_flag[0] = 1
    return 0
#################### A job handler ######################################
def jobHandler(jobId, split, factor, batch_size, samples, status, exit_time ):
    ## call train.py using (split, factor, batch_size, samples)
    ## On Completion:
    time.sleep(3)  ## REMOVE THIS AND ADD ACTUAL JOB
    status[0] = 1
    exit_time[0] = time.time()
    return 0

#################### Main #################################################
if __name__ == "__main__":


    ## Lets the job come in!
    job_server = threading.Thread(target=job_arrivals, args=(rate,T_exp, ))
    job_server.start()


    job_served      = 0  # no of jobs that have started running
    jobs_in_process = []
    jobs_completed  = [] 
    no_jobs_waiting = [] # number of jobs waiting in queue

    t = time.time()
    while not end_exp_flag[0]:
        try: 
            ## Check if a new job arrived. Will fail if no new job
            new_job = job_queue[job_served+1] 

            if (new_job['entry_time'][0]):
                print("Something went wrong! New job already has entry time")
            
            ## get a resource allocation for new job
            if algo == 'baseline':
                split = get_split_algo_1()
            elif algo == 'bigbrain':
                split = get_split_algo_2()
            else:
                print("Invalid algo!")
            
            ## If resource is available then a split is returned 
            ## Else just wait
            if split:   
                new_job['model_split'] = split
                ### Update resources
                update_resource(split, 'acquire')
                if resource_manager[0]<0 or resource_manager[1]<0:
                    print("Error! There's nothing like negative resources!")
                
                ### Spawn a thread to start the new job
                new_job['entry_time'][0]  = time.time()
                job_submit = threading.Thread(target=jobHandler, args=(job_served+1, new_job['model_split'], fct, \
                                                                      batch_size, Nrun, new_job['status_flag'], \
                                                                      new_job['exit_time'], ))
                job_submit.start()
                job_served = job_served+1
                jobs_in_process.append(job_served)
                
            else:
                pass
            
            
        except:
            pass
        
        completed = []
        for inprocess_job_id in jobs_in_process:
            if job_queue[inprocess_job_id]['status_flag'][0]:
                completed.append(inprocess_job_id)   
                ## Release resources
                update_resource(job_queue[inprocess_job_id]['model_split'], 'release')
        
        jobs_completed = jobs_completed +  completed 
        for i in completed:
            jobs_in_process.remove(i)
            
        t_now = time.time()
        if t_now-t>1:
            ## Check for completed jobs 
            no_waiting = len(job_queue)-job_served
            no_jobs_waiting.append(no_waiting)
            print('*'*20)
            print("Jobs in process:", len(jobs_in_process))
            print("Jobs waiting:",no_waiting )
            print("Jobs Completed: ", len(jobs_completed))
            print('*'*20)
            t = t_now
            

        time.sleep(0.1)

    #### Post processing 
    #plt.plot(no_jobs_waiting)
    #plt.show()
    print(no_jobs_waiting)

    process_time  = []
    waiting_time = []
    for job in job_queue:
        if job_queue[job]['status_flag'][0]:
            waiting_time.append(job_queue[job]['entry_time'][0] - job_queue[job]['arrival_time'][0])
            process_time.append(job_queue[job]['exit_time'][0] - job_queue[job]['entry_time'][0])

    print("Mean Waiting Time: ", np.mean(waiting_time))
    print("Mean Process Time: ", np.mean(process_time))
 
