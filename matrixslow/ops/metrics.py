# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:34:46 CST 2019

@author: chenzhen
"""

import numpy as np
from core import Node


class Metrics(Node):
    pass


class Accuracy(Metrics):
    def __init__(self, *parents):
        Node.__init__(self, *parents)
        self.correct_num = 0
        self.total_num = 0

    def compute(self):
        '''
        计算Accrucy: (TP + TN) / TOTAL
        这里假设第一个父节点是预测值（概率），第二个父节点是标签
        '''
        pred = np.where(self.parents[0].value < 0.5, 0, 1)
        gt = self.parents[1].value
        assert len(pred) == len(gt)
        self.correct_num += np.sum(pred == gt)
        self.total_num += len(pred)
        if self.total_num != 0:
            self.value = float(self.correct_num) / self.total_num

    def get_jacobi(self):
        raise NotImplementedError()
