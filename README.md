# Big Brain: Efficient Cluster Management for Deep Learning Jobs

With the rising popularity of large scale Deep Learning (DL) workload, GPU clusters have become a necessity. Yet, the tools to manage them are far from ideal. On one hand the deep learning engineers are burdened with task of estimating resource management usage of their models and routinely face Out-of-memory (OOM) errors. On the other hand, typ- ical GPU cluster utilization are low, often below 30%. But with recent developments in GPU virtualization and drawing from our learnings building Beachi - a fast model splitting system - we explore ways to improve GPU cluster manage- ment for DL. Model Parallelism (MP) techniques have so far been seen as a need, when the models are too big. However, we (plan to) demonstrate MP as an essential tool in building efficient DL workflows.




Detailed discussions can be found here: https://www.notion.so/CS523-Big-Brain-Model-Parallelism-over-distributed-shared-infrastructure-c9c71de6640847aa8a81e4f78910efcd


# Repo

Repo consists of two parts: The BigBrain Kubernetes operator (as shown in the demo video) and the experiments as jupyter notebooks(as described in the report)

Create the following conda env to run the experiments:
'''
# Run:
conda create --name pytorch_baechi python=3.6
conda activate pytorch_baechi

conda install -y \
      python=3.6 \
      numpy=1.16 \
      bazel=0.20.0 \
      networkx \
      future \
      matplotlib \
      cvxopt \
      scikit-learn \
      cudatoolkit=10.0 \
      cudnn \
      cupti

pip install -f https://download.mosek.com/stable/wheel/index.html Mosek==8.1.82


***For running pytorch/summarize.py (torch-1.9.0)***


conda install pytorch torchvision torchaudio -c pytorch # torch 1.9 (as of 20 July 2021)
conda install -c conda-forge jupyterlab
pip install sklearn matplotlib opencv-python
python3 summarize.py --prof_rounds 4 --prof_gpu_id 0 --gpu_num 4 --sch sct --batch_size 32 --type all


***To track GPU usage***


pip install GPUtil


***To plot memory traces (not needed for the core code)***


pip install pandas


***for gpu memory commands: torch.cuda.list_gpu_processes()***


***For debug***

pip install psutil

'''