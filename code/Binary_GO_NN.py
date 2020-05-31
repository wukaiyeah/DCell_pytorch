import sys
import os
import math
import torch
from torch import nn
from config import Config
from net_layers import BranchNode,RootNode
from copy import deepcopy

def get_neuron_num(term_list,term_size_map):
    neuron_size_map = {}
    term_state = {}
    for i,term_name in enumerate(term_list):
        term_state[term_name] = 0
        neuron_size_map[term_name] = max(10, math.floor(0.2*term_size_map[term_name]))
    return neuron_size_map,term_state

def generate_mask_node_map(term_list, children_gene_map, neuron_size_map, gene2id, feature_dim):
    mask_node_map = {}
    gene_node_map = {}
    for i,term_name in enumerate(term_list):
        children_gene_list = children_gene_map[term_name]
        if len(children_gene_list) != 0:
            linear_layer = nn.Linear(feature_dim,neuron_size_map[term_name]) # 后续应该删除
            gene_node_map[term_name] = linear_layer

            term_mask = torch.zeros(feature_dim,neuron_size_map[term_name]) # 生成0的矩阵
            for j,gene_name in enumerate(children_gene_list):
                term_mask[gene2id[gene_name]] = 1
        mask_node_map[term_name] = term_mask.transpose(0,1) # 制作mask, 标记该GO_term下基因的位置
    return mask_node_map, gene_node_map

#print("Constructing NN term --> combine gene --> batch normalization --> hard tanh --> residue --> binary")

def filter_terms(term_state,children_term_map): # 用于将GO_term分层，从底层开始迭代返回Term
    result_list = []
    for term_name,state in zip(term_state.keys(), term_state.values()):        
        if state == 0:
            children_term_list = children_term_map[term_name]
            child_all_ready = True

            for i,child in enumerate(children_term_list):
                if term_state[child] == 0: #检测子term的state若是0，则不写入result_list
                    child_all_ready = False
                    break

            if child_all_ready == True:
                result_list.append(term_name)
    return result_list



def make_node_layers(term_state, children_term_map, children_gene_map, neuron_size_map, feature_dim):
    '''
    construct net layers for each node GO_term
    为每个节点分别建立layers
    '''
    term_states = deepcopy(term_state)
    term_node_layers = {}
    opt = Config() 

    ready_term = filter_terms(term_states,children_term_map) # 从最低层节点 返回同一级别GO_term名
    while len(ready_term) != 0: #root节点的数量是1,从最底层节点循环至root节点
        ready_term = filter_terms(term_states,children_term_map)
        print("New round selects",len(ready_term),"terms")

        for i,term_name in enumerate(ready_term):
            children_term_list = children_term_map[term_name]
            children_gene_list = children_gene_map[term_name]
            hidden_num1 = neuron_size_map[term_name]
            hidden_num2 = 0
            #--计算hidden_num2的值:为本节点layer1的输出+子节点的layer2输出
            # Combine term hidden and direct gene input
            if len(children_term_list) > 0:
                for j,child in enumerate(children_term_list):
                    if term_node_layers[child] == None: # 检查该term_name的子term有无建立网络，按理说此时最低级的term已经构建完net
                        print("Logical mistakes happen here",child,"missing!")
                    hidden_num2 = hidden_num2 + neuron_size_map[child] # 此term的所有子term的输出相加
              
                #if len(children_gene_list) != 0: # 若本节点独有基因数目不为0，则此层layer1输出有值
                hidden_num2 = hidden_num2 + neuron_size_map[term_name]# 子term的hidden_num1 + 本term的hidden_num1
                #else:
                #    hidden_num2 = hidden_num2
            else:
                hidden_num2 = neuron_size_map[term_name] # 最底层节点的hidden_num2无子节点输入

            out_num = neuron_size_map[term_name]

            if term_name != opt.root: # 若为其余 子term
                term_node_layers[term_name] = BranchNode(feature_dim, hidden_num1, hidden_num2, out_num)
            else: # 若为根term
                term_node_layers[term_name] = RootNode(feature_dim, hidden_num1, hidden_num2, out_num)
            term_states[term_name] = 1 # 构建完此term_name,更改标记
    return term_node_layers

class CombineNode(nn.Module):
    '''
    将每个节点建立网络按照顺序串联起来，构建整个前向传播网络
    Construct the whole GO_node net-layers for forward propagation
    '''
    def __init__(self, term_node_layers, children_term_map, term_state):
        super(CombineNode,self).__init__()
        self.term_node_layers = term_node_layers
        self.children_term_map = children_term_map
        self.term_state = term_state
    

    def forward(self, x): 
        trans_out = {}
        predict_out = {} # dict放每个节点的预测值
        opt = Config() # 加载参数
        term_states = deepcopy(self.term_state)

        ready_term = filter_terms(term_states, self.children_term_map) # 从最低层节点 返回同一级别GO_term名
        while len(ready_term) != 0: #root节点的数量是1
            ready_term = filter_terms(term_states, self.children_term_map)
            # print("New round selects",len(ready_term),"terms")

            for i,term_name in enumerate(ready_term):
                children_term_list = self.children_term_map[term_name]
                node_layer = self.term_node_layers[term_name]
                previous_out_list = [] # 每次清空list
                if len(children_term_list) > 0:
                    for j,child in enumerate(children_term_list):
                        if self.term_node_layers[child] == None: # 检查该term_name的子term有无建立网络，按理说此时最低级的term已经构建完net
                            print("Logical mistakes happen here",child,"missing!")
                        previous_out_list.append(trans_out[child])

                    previous_out = torch.cat(previous_out_list, 1) # 合并list中所有tensor成一个tensor
                    trans_out[term_name], predict_out[term_name] = node_layer(x, previous_out) # 最底层节点数据输入仅需要training数据
                else:
                    previous_out = None
                    trans_out[term_name], predict_out[term_name] = node_layer(x, previous_out) # 最底层节点数据输入仅需要training数据

                term_states[term_name] = 1 # 构建完此term_name,更改标记
        return predict_out