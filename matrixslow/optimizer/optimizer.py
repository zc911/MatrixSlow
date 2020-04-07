# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:25:06 2019

@author: zhangjuefei
"""
import abc

import numpy as np

from ..core import Node, Variable, get_node_from_graph
from ..core.graph import Graph


class Optimizer(object):
    """
    优化器基类
    """

    def __init__(self, graph, target, learning_rate=0.01):
        """
        优化器的构造函数接收一个计算图，计算图中的目标节点，以及学习率超参
        """
        assert isinstance(target, Node) and isinstance(graph, Graph)
        self.graph = graph
        self.target = target
        self.learning_rate = learning_rate

        # 为每个参与训练的节点累加一个Mini Batch的全部样本的梯度
        self.acc_gradient = dict()
        self.acc_no = 0

    def one_step(self):
        """
        计算并累加样本的梯度
        """
        self.forward_backward()
        self.acc_no += 1

    def get_gradient(self, node):
        """
        返回样本的平均梯度
        """
        assert node in self.acc_gradient
        return self.acc_gradient[node] / self.acc_no

    @abc.abstractmethod
    def _update(self):
        '''
        抽象方法，执行具体的梯度更新算法，由子类实现
        '''

    def apply_gradients(self, node_gradients_dict, summarize=False, acc_no=None):
        for node, gradient in node_gradients_dict.items():
            if isinstance(node, Node):
                pass
            else:
                target_node = get_node_from_graph(node)
                assert target_node is not None
                assert self.acc_gradient[target_node].shape == gradient.shape
                if summarize:
                    self.acc_gradient[target_node] += gradient
                else:
                    self.acc_gradient[target_node] = gradient
        if summarize:
            self.acc_no += acc_no
        else:
            if acc_no is None:
                # 传入的是平均梯度, 强制让acc_no变为1，避免梯度更新时重复平均
                self.acc_no = 1
            else:
                self.acc_no = acc_no

    def update(self, var_gradients=None):
        if var_gradients is not None:
            self.apply_gradients(var_gradients)
        self._update()
        # 清除累计梯度
        self.acc_gradient.clear()
        self.acc_no = 0

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


class GradientDescent(Optimizer):
    """
    梯度下降优化器
    """

    def __init__(self, graph, target, learning_rate=0.01):
        Optimizer.__init__(self, graph, target)
        self.learning_rate = learning_rate

    def _update(self):
        """
        利用梯度更新可训练变量
        """
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                gradient = self.get_gradient(node)

                node.set_value(node.value - self.learning_rate * gradient)


class Momentum(Optimizer):
    '''
    Momentum动量梯度下降
    '''

    def __init__(self, graph, target, learning_rate=0.01, momentum=0.9):
        Optimizer.__init__(self, graph, target)
        self.learning_rate = learning_rate
        # 动量参数，默认为0.9
        self.momentum = momentum

        self.v = dict()

    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                gradient = self.get_gradient(node)

                if node not in self.v:
                    self.v[node] = gradient
                else:
                    # 动量参数乘以历史
                    self.v[node] = self.momentum * self.v[node] \
                        + self.learning_rate * gradient
                node.set_value(node.value - self.v[node])


class AdaGrad(Optimizer):
    '''
    AdaGrad优化器
    '''

    def __init__(self, graph, target, learning_rate=0.01):
        Optimizer.__init__(self, graph, target)
        self.learning_rate = learning_rate

        self.s = dict()

    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                gradient = self.get_gradient(node)
                # 保存梯度的平方和
                if node not in self.s:
                    self.s[node] = np.power(gradient, 2)
                else:
                    self.s[node] = self.s[node] + np.power(gradient, 2)
                # 通过求历史平方和的开方的倒数，动态调整学习率
                node.set_value(node.value - self.learning_rate *
                               gradient / (np.sqrt(self.s[node] + 1e-10)))


class RMSProp(Optimizer):
    """
    RMSProp优化器
    """

    def __init__(self, graph, target, learning_rate=0.01, beta=0.9):
        Optimizer.__init__(self, graph, target)
        self.learning_rate = learning_rate
        # 新的超参 beta
        assert 0.0 < beta < 1.0
        self.beta = beta

        self.s = dict()

    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                gradient = self.get_gradient(node)

                if node not in self.s:
                    self.s[node] = np.power(gradient, 2)
                else:
                    # 通过 beta 超参求加权平方
                    self.s[node] = self.beta * self.s[node] + \
                        (1 - self.beta) * np.power(gradient, 2)
                # 与AdaGrad完全一致
                node.set_value(node.value - self.learning_rate *
                               gradient / (np.sqrt(self.s[node] + 1e-10)))


class Adam(Optimizer):
    """
    Adam优化器
    """

    def __init__(self, graph, target, learning_rate=0.01, beta_1=0.9, beta_2=0.99):
        Optimizer.__init__(self, graph, target)
        self.learning_rate = learning_rate

        assert 0.0 < beta_1 < 1.0
        self.beta_1 = beta_1

        assert 0.0 < beta_2 < 1.0
        self.beta_2 = beta_2
        # 一阶梯度累积
        self.v = dict()
        # 二阶梯度累积
        self.s = dict()

    def _update(self):

        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                gradient = self.get_gradient(node)

                if node not in self.s:
                    self.v[node] = gradient
                    self.s[node] = np.power(gradient, 2)
                else:
                    # 梯度累积
                    self.v[node] = self.beta_1 * self.v[node] + \
                        (1 - self.beta_1) * gradient
                    self.s[node] = self.beta_2 * self.s[node] + \
                        (1 - self.beta_2) * np.power(gradient, 2)
                # 更新参数
                node.set_value(node.value - self.learning_rate *
                               self.v[node] / np.sqrt(self.s[node] + 1e-10))
