## Introduction:
This is the code for the ICLR 2024 paper: [Consistency Training with Learnable Data Augmentation for Graph Anomaly Detection with Limited Supervision.](https://openreview.net/forum?id=elMKXvhhQ9)

In this work, we propose a novel framework, ConsisGAD, which is tailored for graph anomaly detection in scenarios characterized by limited supervision and is anchored in the principles of consistency training. Under limited supervision, ConsisGAD effectively leverages the abundance of unlabeled data for consistency training by incorporating a novel learnable data augmentation mechanism, thereby introducing controlled noise into the dataset. Moreover, ConsisGAD takes advantage of the variance in homophily distribution between normal and anomalous nodes to craft a simplified GNN backbone, enhancing its capability to effectively distinguish between these two classes. A brief overview of our framework is illustrated in the following picture.

<p align="center">
  <img src="framework.png" alt="Overall framework of ConsisGAD.">
</p>

This repository contains the source code for our Graph Neural Network (GNN) backbone, consistency training procedure, and learnable data augmentation module. Below is an overview of the key components and their locations within the repository:

- **GNN Backbone Model**: The core implementation of our GNN backbone model is encapsulated within the `simpleGNN_MR` class located in the `models.py` file.

- **Consistency Training Procedure**: The consistency training procedure is implemented through the `UDA_train_epoch` function, which can be found in the `main.py` file.

- **Learnable Data Augmentation**: Our learnable data augmentation is realized via the `SoftAttentionDrop` class, which is also located in the `main.py` file.

## Directory Structure
The repository is organized into several directories, each serving a specific purpose:

- `data/`: This directory houses the datasets utilized in our work.

- `config/`: This folder stores the hyper-parameter configuration of our model.

- `modules/`: Auxiliary components of our model are stored in this directory. It includes important modules, such as the data loader `data_loader.py` and the evaluation pipeline `evaluation.py`.

- `model-weights/`: Here, we store the trained weights of our model.

## Installation:
- Install required packages: `pip install -r requirements.txt` 
- Dataset resources:
    - For Amazon and YelpChi, we use the built-in datasets in the DGL package https://docs.dgl.ai/en/0.8.x/api/python/dgl.data.html.
    - For T-Finance and T-Social, we download the datasets from https://github.com/squareRoot3/Rethinking-Anomaly-Detection. 
    - Please download and unzip all the files in the `data/` folder.

## Usage:
- Hyper-parameter settings for all datasets are put into the `config/` folder.
- To run the model, use `--config` to specify hyper-parameters and `--runs` the number of running times.
- If you want to run the YelpChi dataset 5 times, please execute this command: `python main.py --config 'config/yelp.yml' --runs 5`.

## Citation
If you find our work useful, please cite:

```
@inproceedings{
chen2024consistency,
title={Consistency Training with Learnable Data Augmentation for Graph Anomaly Detection with Limited Supervision},
author={Nan Chen and Zemin Liu and Bryan Hooi and Bingsheng He and Rizal Fathony and Jun Hu and Jia Chen},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=elMKXvhhQ9}
}
```
