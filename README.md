# Graph Spatial Information Preservation (GSIP)
This is the official repository for the paper [What Matters in Graph Class Incremental Learning? An Information Preservation Perspective].

 

 ## Get Started
 
 This repository contains our GSIP implemented for running on GPU devices. To run the code, the following packages are required to be installed:
 
* python==3.7.10
* scipy==1.5.2
* numpy==1.19.1
* torch==1.7.1
* networkx==2.5
* scikit-learn~=0.23.2
* matplotlib==3.4.1
* ogb==1.3.1
* dgl==0.6.1
* dgllife==0.2.6



# Usage

Below is the example to run the ERGNN baseline with GCN backbone on the CoraFull-CL and Reddit-CL datasets under the class-IL scenario. 
 
```
 python train.py --dataset CoraFull-CL \
        --n_base 2 \ 
        --n_cls_per_task 2 \ 
        --ergnn_args="'budget':[100];'d':[0.5];'sampler':['CM'];'w_ll':[50];'w_lg':[0.05];'w_h':[10]" \
        --neibt1=0.5 \ 
        --method ergnn \
        --backbone GCN \
        --gpu 0 \
        --ILmode classIL \
        --inter-task-edges False \
 ```
 ```
 python train.py --dataset Reddit-CL \
        --n_base 10 \ 
        --n_cls_per_task 5 \ 
        --ergnn_args="'budget':[100];'d':[0.5];'sampler':['CM'];'w_ll':[1];'w_lg':[1e-3];'w_h':[5e-6]" \
        --neibt1=0.9 \ 
        --method ergnn \
        --backbone GCN \
        --gpu 0 \
        --ILmode classIL \
        --inter-task-edges False \
 ```



# Cite
If you find this repo useful, please cite

```
@inproceedings{GSIP,
  author    = {Jialu Li and Yu Wang and Pengfei Zhu and Wanyu Lin and Qinghua Hu},
  title     = {What Matters in Graph Class Incremental Learning? An Information Preservation Perspective},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2024}
  pages     = {1-14},
}
```

# Credit
This repository was developed based on the [CGLB](https://github.com/QueuQ/CGLB) and [CaT](https://github.com/superallen13/CaT-CGL).
 
 

