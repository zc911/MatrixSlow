# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:25:06 2019

@author: zhangjuefei
"""

from core.node import *


class Optimizer:
    """
    优化器基类
    """

    def __init__(self, graph, target, batch_size=12):
        assert isinstance(target, Node) and isinstance(graph, Graph)
        self.graph = graph
        self.target = target
        self.batch_size = batch_size

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
        抽象方法，利用梯度更新可训练变量
        """

        pass

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

    def __init__(self, graph, target, learning_rate=0.01, batch_size=12):
        Optimizer.__init__(self, graph, target, batch_size)
        self.learning_rate = learning_rate

    def update(self):
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

    def __init__(self, graph, target, learning_rate=0.01, momentun=0.9, batch_size=32):
        Optimizer.__init__(self, graph, target, batch_size)
        self.learning_rate = learning_rate
        # 动量参数，默认为0.9
        self.momentum = momentun

        self.v = dict()

    def update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                gradient = self.get_gradient(node)

                if node not in self.v:
                    self.v[node] = gradient
                else:
                    self.v[node] = self.momentum * \
                        self.v[node] + self.learning_rate * gradient
                node.set_value(node.value - self.v[node])


class AdaGrad(Optimizer):
    '''
    AdaGrad优化器
    '''

    def __init__(self, graph, target, learning_rate=0.01, beta=0.9, batch_size=32):
        Optimizer.__init__(self, graph, target, batch_size)
        self.learning_rate = learning_rate
        self.beta = 0.9

        self.s = dict()

    def update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                gradient = self.get_gradient(node)

                if node not in self.s:
                    self.s[node] = np.power(gradient, 2)
                else:
                    self.s[node] = self.s[node] + np.power(gradient, 2)

                node.set_value(node.value - self.learning_rate *
                               gradient / (np.sqrt(self.s[node] + 1e-10)))


class RMSProp(Optimizer):
    """
    RMSProp优化器
    """

    def __init__(self, graph, target, learning_rate=0.01, beta=0.9, batch_size=32):
        Optimizer.__init__(self, graph, target, batch_size)
        self.learning_rate = learning_rate

        assert 0.0 < beta < 1.0
        self.beta = beta

        self.s = dict()

    def update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                gradient = self.get_gradient(node)

                if node not in self.s:
                    self.s[node] = np.power(gradient, 2)
                else:
                    self.s[node] = self.beta * self.s[node] + \
                        (1 - self.beta) * np.power(gradient, 2)

                node.set_value(node.value - self.learning_rate *
                               gradient / (np.sqrt(self.s[node] + 1e-10)))


class Adam(Optimizer):
    """
    Adam优化器
    """

    def __init__(self, graph, target, learning_rate=0.01, beta_1=0.9, beta_2=0.99, batch_size=32):
        Optimizer.__init__(self, graph, target, batch_size)
        self.learning_rate = learning_rate

        assert 0.0 < beta_1 < 1.0
        self.beta_1 = beta_1

        assert 0.0 < beta_2 < 1.0
        self.beta_2 = beta_2

        self.s = dict()
        self.v = dict()

    def update(self):

        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                gradient = self.get_gradient(node)

                if node not in self.s:
                    self.v[node] = gradient
                    self.s[node] = np.power(gradient, 2)
                else:
                    self.v[node] = self.beta_1 * self.v[node] + \
                        (1 - self.beta_1) * gradient
                    self.s[node] = self.beta_2 * self.s[node] + \
                        (1 - self.beta_2) * np.power(gradient, 2)

                node.set_value(node.value - self.learning_rate *
                               self.v[node] / np.sqrt(self.s[node] + 1e-10))
