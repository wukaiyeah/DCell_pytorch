import sys
import os
import torch
from torch import nn

class BranchNode(nn.Module):
    '''
    Full connect layer for the children GO_term
    全连接层，用于非root GO_term节点的数据处理层
    '''
    def __init__(self, feature_dim, hidden_num1, hidden_num2, out_num):
        super(BranchNode,self).__init__()
        self.layer1 = nn.Linear(feature_dim, hidden_num1)
        self.layer2 = nn.Sequential(
                    nn.Linear(hidden_num2, out_num),
                    nn.BatchNorm1d(out_num), # 或者为 nn.BatchNorm2d(hidden_num) 
                    nn.Tanh() # 激活函数
        )
        self.layer3 = nn.Sequential(
            nn.Linear(out_num,1),
            nn.Tanh()
        )

    def forward(self, x, previous_out = None): # previous_out应该是一个包含多个tensor的list，包含多个子节点输入
        x = self.layer1(x)
        
        if previous_out != None: # 上层节点的输出在此合并，一同输入
            x = torch.cat((x, previous_out),1)
        else:
            x = x
        
        trans_out = self.layer2(x) # 传入下层节点
        predict_out = self.layer3(trans_out) # 输出该节点的预测值
        return trans_out, predict_out



class RootNode(nn.Module):
    '''
    Full connect layer for the root GO_term
    全连接层，用于Root级GO Term，其最终输出为一个数
    '''
    def __init__(self, feature_dim, hidden_num1, hidden_num2, out_num):
        super(RootNode,self).__init__()
        self.layer1 = nn.Linear(feature_dim, hidden_num1)
        self.layer2 = nn.Sequential(
                    nn.Linear(hidden_num2, out_num), 
                    nn.BatchNorm1d(out_num),
                    nn.Tanh(),
                    nn.Linear(out_num,1)
        )
    
    def forward(self, x, previous_out = None): 
        x = self.layer1(x)

        if previous_out != None: # 判断有无
            x = torch.cat((x, previous_out),1)
        else:
            x = x

        trans_out = None
        predict_out = self.layer2(x)       
        return trans_out, predict_out
