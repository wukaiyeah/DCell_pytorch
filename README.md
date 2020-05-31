# DCell_pytorch
## Cell Pytorch version base on DCell_torch7(Lua) from 
https://github.com/idekerlab/DCell
For more infomation, please refer to the original code, research publication, dataset, et al.
This pytorch version of DCell is only for pytorch self-study.

## 1.Directory
Code/ 'All script for DCell demo'
Topology/ 'The demo GO_term'
TrainData/ 'The experiment data(Knock-down gene, and cell viability change values)'
model/ 'The directory for saved training models, empty now'

## 2.Dependence:System & Hardware
System: Ubuntu-18.04
GPU:NVIDIA-GeForce-GTX 1650
    No GPU or Higher performance GPU is Ok for demo dataset
RAM: 16GB of my computer (8GB at least)

## 3.Requirement
python: 3.7.6
pytorch: 1.4.0
numpy: 1.18.1

## 4.Demo Training
4.1 Change datapath in code/config.py 
4.2 For training: `python Train_DCell.py`
4.3 For testing(genarete model): `python Test_DCell.py`
