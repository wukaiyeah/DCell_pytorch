import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn

def load_ontology_file(topo_file):
    assert os.path.exists(topo_file), 'Can not find ontology file!'

    children_term_map = {}
    children_gene_map = {}
    term_size_map = {}
    term_list = []
    gene_list = []

    rfile = open(topo_file, 'r')
    for line in rfile.readlines():
        if line.startswith('ROOT'):
            terms = line.strip().split()
            term_name = terms[1]
            term_list.append(term_name) # ROOT中的GO_ID取出
            term_size_map[term_name] = int(terms[2])

        if line.startswith('GENES'):
            terms = line.strip().split()
            children_gene_list = terms[1:]
            gene_list += children_gene_list
            children_gene_map[term_name] = children_gene_list

        if line.startswith('TERMS'):
            terms = line.strip().split()
            children_term_list = terms[1:]
            children_term_map[term_name] = children_term_list
    
    term_list = list(set(term_list)) # remove duplicates
    gene_list = list(set(gene_list)) 

    rfile.close()
    return term_list, gene_list, children_term_map, children_gene_map, term_size_map

def cal_gene2id(gene_list):
    gene2id = {}
    for id,gene in enumerate(gene_list):
        gene2id[gene] = id
    return gene2id

def load_GI_file(GI_file):
    assert os.path.exists(GI_file), 'Can not find training data file!'
    
    genotype_list = []
    phenotype_list = []
    max_KO_num = -1

    rfile = open(GI_file, 'r')
    for line in rfile.readlines():
        items = line.strip().split()
        KO_num = int(items[0])# number of knock-out genes,敲除基因数目
        if KO_num > max_KO_num: # record the max_KO 记录最大敲除基因数目
            max_KO_num = KO_num

        genotype = {}
        for i in range(1, KO_num + 1):
            genotype[i] = items[i]
        genotype_list.append(genotype) # genotype

        if len(items) > KO_num+1:
            phenotype_list.append(float(items[-1])) # collect cell viability values into phenotype,将KO后的细胞分裂能力数值放入phenotype

    rfile.close()
    return genotype_list, phenotype_list, max_KO_num

def list2torch(genotype_list, phenotype_list, gene2id, max_KO):
    genotype_th = torch.ones((len(genotype_list),max_KO+1)) * -1 # construct a -1 tensor
    phenotype_th = torch.zeros(len(genotype_list))

    for i,genotype in enumerate(genotype_list):
        genotype_th[i][0] = len(genotype) # 第一个位置写入每条genotype数目
        for j,gene in enumerate(genotype.values()):
            if gene not in gene2id.keys():
                print('Gene',gene,'does not have index...')
                continue
            else:
                genotype_th[i][j+1] = gene2id[gene] # 第二、三个位置写入基因ID

        phenotype_th[i] = phenotype_list[i] # 这里phenotype没有做变化
    return genotype_th, phenotype_th

def expand_KO(phenotype_th,feature_num):
    '''
    将基因敲除实验数据，转化为0/1的tensor，0为wild type基因位置，1为mutation基因位置
    '''
    feature = torch.zeros(phenotype_th.shape[0],feature_num)

    for i in range(phenotype_th.shape[0]):
        for j,num in enumerate(phenotype_th[i][1:]):
            feature[i][int(num)] = 1

    return feature

class PrepareDataset(Dataset):
    '''
    prepare dataset for dataloader
    '''
    # Initialize your datasets, labels, etc.
    def __init__(self, datasets, labels):
        # 加载datasets
        self.x_data = datasets
        # 加载labels
        self.y_data = labels
        # 获取长度
        self.len = datasets.shape[0]
        
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
        
    def __len__(self): 
        # 返回文件数据的数目
        return self.len

class LossCompute():
    '''
    Compute the loss of MSE by weight of each GO_term
    每个节点先计算MSE，再按不同权重(root:1, others:0.3)对每个节点校准，再加和
    '''
    def __init__(self, root_term):
        self.root_term = root_term

    def finalLoss(self, outputs, labels):
        MSE = nn.MSELoss()
        Loss_list = []
        for term_name,predict in zip(outputs.keys(),outputs.values()):
            if term_name == self.root_term:
                Loss_list.append(MSE(predict.view(-1), labels)) # .view(-1)更改维度，与labels一致
            else:
                Loss_list.append(MSE(predict.view(-1), labels)*0.3) # 非root节点需 *0.3加权
        return sum(Loss_list)

def paramsInit(term_node_layers, mask_node_map):
    for term_name,term_layer in zip(term_node_layers.keys(), term_node_layers.values()): # 每个节点循环处理
        grad_mask = mask_node_map[term_name]
        if grad_mask == None: # 判断mask是否存在
            print('There is no mask for term',term_name)
        if term_layer.layer1.weight.shape != grad_mask.shape: # 判断二者维度是否一致，但grad_mask是个向量就可，无必要维度一致的矩阵
            print("Gradient mistakes, size not match!",term_name)
        term_layer.layer1.weight.data *= grad_mask # 第一层layer1的weight，用mask处理

def make_params_list(term_node_layers):
    params_list = []
    params_dict = {}
    params = {}
    for term_name,term_layer in zip(term_node_layers.keys(), term_node_layers.values()):
        params_dict['params'] = term_layer.parameters()
        params_list.append(params_dict)
    params['params'] = params_list
    return params