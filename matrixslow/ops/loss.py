
# -*- coding: utf-8 -*-
import numpy as np
from core.node import Node


class LogLoss(Node):
    """
    LogLoss函数
    """

    def compute(self):
        y_pred = self.parents[0]
        y_gt = self.parents[1]
        assert y_pred.shape() == y_gt.shape()
        eps = 1e-15
        prob = np.clip(y_pred.value, eps, 1 - eps)
        self.value = np.mat(np.sum(- np.multiply(y_gt.value, np.log(prob)) -
                                   np.multiply((1 - y_gt.value), np.log(1 - prob))) / len(y_pred.value))

    # TODO
    def get_jacobi(self, parent):
        prob = SoftMax.softmax(self.parents[0].value)
        if parent is self.parents[0]:
            return (prob - self.parents[1].value).T
        else:
            return (-np.log(prob)).T


class SoftMax(Node):
    """
    SoftMax函数
    """

    @staticmethod
    def softmax(a):
        a[a > 1e2] = 1e2  # 防止指数过大
        ep = np.power(np.e, a)
        return ep / np.sum(ep)

    def compute(self):
        self.value = SoftMax.softmax(self.parents[0].value)

    def get_jacobi(self, parent):
        """
        我们不实现SoftMax节点的get_jacobi函数，训练时使用CrossEntropyWithSoftMax节点（见下）
        """
        return np.mat(np.eye(self.dimension()))  # 无用


class CrossEntropyWithSoftMax(Node):
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
