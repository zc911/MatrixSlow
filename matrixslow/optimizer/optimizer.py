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
        优化器的构造函数接受计算图对象，目标节点对象以及学习率
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
        """
        抽象方法，执行具体的梯度更新算法，由子类实现
        """

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

        # 执行更新
        self._update()

        # 清除累加梯度
        self.acc_gradient.clear()
        self.acc_no = 0

    def forward_backward(self):
        """
        前向传播计算结果节点的值并反向传播计算结果节点对各个节点的雅可比矩阵
        """

        # 清除计算图中所有节点的雅可比矩阵
        self.graph.clear_jacobi()

        # 前向传播计算结果节点
        self.target.forward()

        # 反向传播计算雅可比矩阵
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                node.backward(self.target)

                # 最终结果（标量）对节点值的雅可比是一个行向量，其转置是梯度（列向量）
                # 这里将梯度reshape成与节点值相同的形状，好对节点值进行更新。
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
        朴素梯度下降法
        """
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:

                # 取得该节点在当前批的平均梯度
                gradient = self.get_gradient(node)

                # 用朴素梯度下降法更新变量节点的值
                node.set_value(node.value - self.learning_rate * gradient)


class Momentum(Optimizer):
    """
    冲量法
    """

    def __init__(self, graph, target, learning_rate=0.01, momentum=0.9):

        Optimizer.__init__(self, graph, target)

        self.learning_rate = learning_rate

        # 衰减系数，默认为0.9
        self.momentum = momentum

        # 积累历史速度的字典
        self.v = dict()

    def _update(self):

        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:

                # 取得该节点在当前批的平均梯度
                gradient = self.get_gradient(node)

                if node not in self.v:
                    self.v[node] = - self.learning_rate * gradient
                else:
                    # 滑动平均累积历史速度
                    self.v[node] = self.momentum * self.v[node] \
                        - self.learning_rate * gradient

                # 更新变量节点的值
                node.set_value(node.value + self.v[node])


class AdaGrad(Optimizer):
    """
    AdaGrad优化器
    """

    def __init__(self, graph, target, learning_rate=0.01):

        Optimizer.__init__(self, graph, target)

        self.learning_rate = learning_rate

        self.s = dict()

    def _update(self):

        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:

                # 取得该节点在当前批的平均梯度
                gradient = self.get_gradient(node)

                # 累积梯度各分量的平方和
                if node not in self.s:
                    self.s[node] = np.power(gradient, 2)
                else:
                    self.s[node] = self.s[node] + np.power(gradient, 2)

                # 更新变量节点的值
                node.set_value(node.value - self.learning_rate *
                               gradient / (np.sqrt(self.s[node] + 1e-10)))


class RMSProp(Optimizer):
    """
    RMSProp优化器
    """

    def __init__(self, graph, target, learning_rate=0.01, beta=0.9):

        Optimizer.__init__(self, graph, target)

        self.learning_rate = learning_rate

        # 衰减系数
        assert 0.0 < beta < 1.0
        self.beta = beta

        self.s = dict()

    def _update(self):

        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:

                # 取得该节点在当前批的平均梯度
                gradient = self.get_gradient(node)

                # 滑动加权累积梯度各分量的平方和
                if node not in self.s:
                    self.s[node] = np.power(gradient, 2)
                else:
                    self.s[node] = self.beta * self.s[node] + \
                        (1 - self.beta) * np.power(gradient, 2)

                # 更新变量节点的值
                node.set_value(node.value - self.learning_rate *
                               gradient / (np.sqrt(self.s[node] + 1e-10)))


class Adam(Optimizer):
    """
    Adam优化器
    """

    def __init__(self, graph, target, learning_rate=0.01, beta_1=0.9, beta_2=0.99):

        Optimizer.__init__(self, graph, target)
        self.learning_rate = learning_rate

        # 历史梯度衰减系数
        assert 0.0 < beta_1 < 1.0
        self.beta_1 = beta_1

        # 历史梯度各分量平方衰减系数
        assert 0.0 < beta_2 < 1.0
        self.beta_2 = beta_2

        # 历史梯度累积
        self.v = dict()

        # 历史梯度各分量平方累积
        self.s = dict()

    def _update(self):

        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:

                # 取得该节点在当前批的平均梯度
                gradient = self.get_gradient(node)

                if node not in self.s:
                    self.v[node] = gradient
                    self.s[node] = np.power(gradient, 2)
                else:
                    # 梯度累积
                    self.v[node] = self.beta_1 * self.v[node] + \
                        (1 - self.beta_1) * gradient

                    # 各分量平方累积
                    self.s[node] = self.beta_2 * self.s[node] + \
                        (1 - self.beta_2) * np.power(gradient, 2)

                # 更新变量节点的值
                node.set_value(node.value - self.learning_rate *
                               self.v[node] / np.sqrt(self.s[node] + 1e-10))
