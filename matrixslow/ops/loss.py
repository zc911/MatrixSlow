# -*- coding: utf-8 -*-
"""
Created on Wed July  9 15:13:01 2019

@author: chenzhen
"""

import numpy as np

from ..core import Node
from ..ops import SoftMax


class LossFunction(Node):
    '''
    定义损失函数抽象类
    '''
    pass


class LogLoss(LossFunction):
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


class CrossEntropyWithSoftMax(LossFunction):
    """
    对第一个父节点施加SoftMax之后，再以第二个父节点为标签One-Hot向量计算交叉熵
    """

    def compute(self):
        prob = SoftMax.softmax(self.parents[0].value)
        self.value = np.mat(
            -np.sum(np.multiply(self.parents[1].value, np.log(prob + 1e-10))))

    def get_jacobi(self, parent):
        # 这里存在重复计算，但为了代码清晰简洁，舍弃进一步优化
        prob = SoftMax.softmax(self.parents[0].value)
        if parent is self.parents[0]:
            return (prob - self.parents[1].value).T
        else:
            return (-np.log(prob)).T


class PerceptionLoss(LossFunction):
    """
    感知机损失，输入为正时为0，输入为负时为输入的相反数
    """

    def compute(self):
        self.value = np.mat(np.where(
            self.parents[0].value >= 0.0, 0.0, -self.parents[0].value))

    def get_jacobi(self, parent):
        """
        雅克比矩阵为对角阵，每个对角线元素对应一个父节点元素。若父节点元素大于0，则
        相应对角线元素（偏导数）为0，否则为-1。
        """
        diag = np.where(parent.value >= 0.0, 0.0, -1)
        return np.diag(diag.ravel())


