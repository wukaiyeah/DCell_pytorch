# DCell_pytorch
## Cell Pytorch version base on DCell_torch7(Lua) from 
https://github.com/idekerlab/DCell<br>
For more infomation, please refer to the original code, research publication, dataset, et al.<br>
This pytorch version of DCell is only for pytorch self-study.<br>

## 1.Directory
Code/ 'All script for DCell demo'<br>
Topology/ 'The demo GO_term'<br>
TrainData/ 'The experiment data(Knock-down gene, and cell viability change values)'<br>
model/ 'The directory for saved training models, empty now'<br>

## 2.Dependence:System & Hardware
System: Ubuntu-18.04<br>
GPU:NVIDIA-GeForce-GTX 1650<br>
    No GPU or Higher performance GPU is Ok for demo dataset<br>
RAM: 16GB of my computer (8GB at least)<br>

## 3.Requirement
python: 3.7.6<br>
pytorch: 1.4.0<br>
numpy: 1.18.1<br>

## 4.Demo Training
4.1 Change datapath in code/config.py <br>
4.2 For training: `python Train_DCell.py`<br>
4.3 For testing(genarete model): `python Test_DCell.py`<br>
