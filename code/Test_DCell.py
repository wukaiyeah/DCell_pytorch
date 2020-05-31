# --------------------------------------------------------
# Pytorch to construct DCell
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import pprint
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import Config # 加载参数
from utility import *
from Binary_GO_NN import *


def main():
    #----------------load config----------------
    print('---------step1.load config--------')
    # global opt
    opt = Config() # load config
    print('Using these config:')
    print(opt.__dict__)

    #---------------- load dataset---------------
    print('---------step2.load & process testing data---------')
    term_list, gene_list, children_term_map, children_gene_map, term_size_map = load_ontology_file(opt.topo)
    print(opt.root, 'ontology contains:')
    print('terms', '\t', len(term_list))
    print('genes', '\t', len(gene_list))

    gene2id = cal_gene2id(gene_list) # transform gene_list into dict['gene':'id']
    feature_dim = len(gene_list) # gene_list数目
    if feature_dim <= 0:
        print('No gene is included..')
        sys.exit()

    # load testing data
    if opt.test != '': # 若opt.test不为空，则可处理
        test_gene_pair_list, test_GI_list, max_KO = load_GI_file(opt.test) 
        # process testing/labeled data
        TestData, TestLabel = list2torch(test_gene_pair_list, test_GI_list, gene2id, max_KO) # TestData:[KO_num, Gene1_id, Gene2_id], TestLabel:[cell_value1, cell_value2,...]
        TestDataExpand = expand_KO(TestData, feature_dim) # 构建基因位置的tensor
        TestDataset = PrepareDataset(TestDataExpand, TestLabel) # 构建dataloader
        TestLoader = DataLoader(dataset= TestDataset, batch_size=8, shuffle=False, num_workers=4, drop_last =True)
        print('Loading testing data',len(TestLabel))
    else:
        print('Can not find the testing data!')
        sys.exit()

    print("Loading data complete......feature dim", feature_dim)

    #---------------- load model---------------
    print('---------step3.construct model layers----------')
    neuron_size_map,term_state = get_neuron_num(term_list,term_size_map) # 计算网络所用神经元数量，最大为10，最大为GO_term下基因数目
    # mask_node_map:每个GO_term的mask(0或1)，每个mask记录了含有基因的位置
    # gene_node_map:构建每个GO_term的linear layer, dict格式
    mask_node_map, gene_node_map = generate_mask_node_map(term_list, children_gene_map, neuron_size_map, gene2id, feature_dim)
    print("gene_node_map size",len(gene_node_map))
    # 为每个节点分别建立layers
    term_node_layers = make_node_layers(term_state, children_term_map, children_gene_map, neuron_size_map, feature_dim)
    print('Layers for each node construct complete')
    # 合并所有节点的网络，拼成完整网络
    net = CombineNode(term_node_layers, children_term_map, term_state)
    print('Whole node layers combind complete')
    if opt.model != '':
        net.load_state_dict(torch.load(opt.model))
    else:
        print('Training model not found!')
    print('All keys matched successfully')
    print('Training model loaded complete')
    print('-----------step4.Setting loss & optimizer---------------')
    paramsInit(term_node_layers, mask_node_map) # 初始化网络参数
    optimizer = optim.Adam([{'params': model.parameters()} for model in term_node_layers.values()], lr = opt.LR, betas = (0.9, 0.99))
    criterion = LossCompute(opt.root)

    print('Loss & optimizer setting complete')
    print('-------------step5.Start Testing----------------------')

    test_predict = []
    test_ground_truth = []
    lossVal = []
    start = time.process_time()
    for j,data in enumerate(TestLoader):
        inputs, labels = data
     
        if torch.cuda.is_available(): # 加载GPU运算
            for term_name,term_layer in zip(term_node_layers.keys(),term_node_layers.values()): # 网络加载GPU
                term_node_layers[term_name] = term_layer.cuda()
            for term_name, grad_mask in zip(mask_node_map.keys(),mask_node_map.values()): # mask加载GPU
                mask_node_map[term_name] = grad_mask.cuda()
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = net(inputs) # forward
        loss = criterion.finalLoss(outputs, labels)

        test_predict.append(outputs[opt.root].data.view(-1))
        test_ground_truth.append(labels)
        lossVal.append(loss)

    corr = np.corrcoef(torch.cat(test_predict,0).tolist(), torch.cat(test_ground_truth,0).tolist())[0,1] # caculate pearson correlation
    MESloss = sum(lossVal)/(TestLabel.shape[0]/opt.batchSize) # caculate total MSE loss of this epoch
    end = time.process_time()

    print("Testing correlation:", corr, 'MSE:', float(MESloss.data), 'Elapsed: %fs'%(end - start))
    print('----------------Testing Complete----------------------')


if __name__ == "__main__":
    main()