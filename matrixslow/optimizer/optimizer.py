# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:25:06 2019

@author: zhangjuefei
"""

from core.node import *


class GradientDescent:
    """
    优化器基类
    """

    def __init__(self, graph, target, learning_rate=0.01, batch_size=12):
        assert isinstance(target, Node) and isinstance(graph, Graph)
        self.graph = graph
        self.target = target
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # 为每个参与训练的节点累加一个Mini Batch的全部样本的梯度
        self.acc_gradient = dict()
        self.acc_no = 0

    def one_step(self):
        """
        计算并累加样本的梯度，一个Mini Batch结束后执行变量更新
        """
        self.forward_backward()

        self.acc_no += 1
        if self.acc_no >= self.batch_size:
            self.update()
            self.acc_gradient.clear()  # 清除梯度累加
            self.acc_no = 0  # 清除计数

    def get_gradient(self, node):
        """
        返回一个Mini Batch的样本的平均梯度
        """
        assert node in self.acc_gradient
        return self.acc_gradient[node] / self.batch_size

    def update(self):
        """
        利用梯度更新可训练变量
        """
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                gradient = self.get_gradient(node)

                node.set_value(node.value - self.learning_rate * gradient)

    def forward_backward(self):
        """
        前向传播计算结果节点的值并反向传播计算结果节点对各个节点的梯度
        """

        # 清除计算图中所有节点的雅可比矩阵
        self.graph.clear_jacobi()

        # 前向传播计算结果节点
        self.target.forward()

        # 反向传播计算梯度
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                node.backward(self.target)

                # 最终结果（标量）对节点值（视作向量）的雅克比是一个行向量，将其转置是梯度（列向量）
                # 这里将梯度 reshape 成与节点值相同的形状，好对节点值进行更新。
                gradient = node.jacobi.T.reshape(node.shape())
                if node not in self.acc_gradient:
                    self.acc_gradient[node] = gradient
                else:
                    self.acc_gradient[node] += gradient
