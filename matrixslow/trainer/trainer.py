# -*- coding: utf-8 -*-

"""
Created on Wed Jul 10 15:19:42 CST 2019

@author: chenzhen
"""
import numpy as np
from core.graph import default_graph
from optimizer import Optimizer
from ops.loss import LossFunction
from ops.metrics import Accuracy
from util import ClassMining
from sklearn.metrics import accuracy_score


class Trainer(object):
    '''
    训练器
    '''

    def __init__(self, input_x, input_y, logits, loss_fn_name, optimizer_name, epoches, eval_on_train=False):
        self.input_x = input_x
        self.input_y = input_y
        self.logits = logits
        self.loss_fn_name = loss_fn_name
        self.optimizer_name = optimizer_name

        self.epoches = epoches
        self.epoch = 0
        self.eval_on_train = eval_on_train

    def setup_graph(self):
        '''
        利用反射机制，实例化具体的损失函数和优化器
        '''
        # 根据名称实例化一个具体的损失函数节点
        self.loss_op = ClassMining.get_instance_by_subclass_name(
            LossFunction, self.loss_fn_name)(self.logits, self.input_y)

        # 根据名称实例化一个具体的优化器实例
        # TODO optimizer parameters
        self.optimizer = ClassMining.get_instance_by_subclass_name(
            Optimizer, self.optimizer_name)(default_graph, self.loss_op)

    def one_step(self, data_x, data_y):
        '''
        执行一次前向计算和一次后向计算(可能)
        '''
        self.input_x.set_value(np.mat(data_x))
        self.input_y.set_value(np.mat(data_y))

        self.optimizer.one_step()

    def eval(self, test_x, test_y):
        '''
        在测试集合上进行算法评估
        '''
        probs = []
        losses = []

        accuracy_op = Accuracy(self.logits, self.input_y)
        for i in range(len(test_x)):
            self.input_x.set_value(np.mat(test_x[i, :]))
            self.input_y.set_value(np.mat(test_y[i, 0]))

            accuracy_op.forward()

        print("Epoch: {:d}，正确率：{:.2f}%".format(
            self.epoch + 1,  accuracy_op.value * 100))

    def main_loop(self, train_x, train_y, test_x, test_y):
        '''
        训练（验证）的主循环
        '''
        for self.epoch in range(self.epoches):
            mini_batch = []
            cur_index = 0
            if self.eval_on_train:
                self.eval(test_x, test_y)
            # TODO fix out boundary bug
            for i in range(len(train_x)):
                self.one_step(train_x[i], train_y[i])

    def train(self, train_x, train_y, test_x, test_y):
        '''
        开始训练(验证)流程
        '''
        assert len(train_x) == len(train_y)
        if test_x is not None and test_y is not None:
            assert len(test_x) == len(test_y)

        # 构建完整的计算图
        self.setup_graph()

        # 传入数据，开始主循环
        self.main_loop(train_x, train_y, test_x, test_y)
