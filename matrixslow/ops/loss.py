# -*- coding: utf-8 -*-
import numpy as np
from core.node import Node


class LogLoss(Node):
    eps = 1e-15

    def compute(self):
        '''
        计算两个父节点的Log Loss
        注意：这里有个硬性假设，第一个父节点是预测值，第二个是label
        '''
        y_pred = self.parents[0]
        y_gt = self.parents[1]
        assert y_pred.shape() == y_gt.shape()
        prob = np.clip(y_pred.value, LogLoss.eps, 1 - LogLoss.eps)
        self.value = np.mat(np.sum(- np.multiply(y_gt.value, np.log(prob)) -
                                   np.multiply((1 - y_gt.value), np.log(1 - prob))) / len(y_gt.value))

    def get_jacobi(self, parent):
        '''
        计算对父节点的雅各比矩阵
        '''
        y_pred = self.parents[0]
        y_gt = self.parents[1]
        prob = np.clip(y_pred.value, LogLoss.eps, 1 - LogLoss.eps)

        if parent is self.parents[0]:
            return np.mat(- (y_gt.value / prob) + (1 - y_gt.value) / (1 - prob))
        else:
            # 默认第二个节点是label，不会参与更新，以下值无用
            return np.mat(-np.log(prob) + np.log(1 - prob))
