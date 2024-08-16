# Expert-Calibrated Learning for Online Optimization with Switching Costs

[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE) [![arxiv](https://img.shields.io/badge/cs.AI-arXiv%3A2204.08572-B31B1B.svg)](https://arxiv.org/abs/2204.08572)

[Pengfei Li](https://www.cs.ucr.edu/~pli081/), [Jianyi Yang](https://jyang-ai.github.io/) and [Shaolei Ren](https://intra.ece.ucr.edu/~sren/)

**Note**

This is the official implementation of the SIGMETRICS 2022 [paper](https://dl.acm.org/doi/10.1145/3530894) 

## Requirements

* python>=3.6

## Installation
* Clone this repo:
```bash
git clone git@github.com:Ren-Research/EC-L2O.git
cd EC-L2O
```
* Install required packages
```bash
pip install -r requirements.txt
```


## Quick start for training
* Run the code
```bash
python train_unroll_batch.py  config_example.json
```
You can visualize the training statistics using tensorboard.

