# -*- coding: utf-8 -*-

"""
Created on Wed Jul 10 15:19:42 CST 2019

@author: chenzhen
"""
import numpy as np

from core.graph import default_graph
from ops.loss import LossFunction
from ops.metrics import Metrics
from optimizer import Optimizer
from util import ClassMining


class Trainer(object):
    '''
    训练器
    '''

    def __init__(self, input_x, input_y, logits,
                 loss_fn_name, optimizer_name,
                 epoches, batch_size=8,
                 eval_on_train=False, metrics_names=None):
        self.input_x = input_x
        self.input_y = input_y
        self.logits = logits
        self.loss_fn_name = loss_fn_name
        self.optimizer_name = optimizer_name
        self.metrics_names = metrics_names

        self.epoches = epoches
        self.epoch = 0
        self.batch_size = 8
        self.eval_on_train = eval_on_train
        self.metrics_ops = []

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
        self.input_x.set_value(np.mat(data_x).T)
        self.input_y.set_value(np.mat(data_y).T)

        self.optimizer.one_step()

    def eval(self, test_x, test_y):
        '''
        在测试集合上进行算法评估
        '''

        # 实例化metrics算子列表
        if self.eval_on_train and len(self.metrics_names):
            self.metrics_ops = []
            for metrics_name in self.metrics_names:
                self.metrics_ops.append(ClassMining.get_instance_by_subclass_name(
                    Metrics, metrics_name)(self.logits, self.input_y))

        # probs = []
        # losses = []
        for i in range(len(test_x)):
            self.input_x.set_value(np.mat(test_x[i]).T)
            self.input_y.set_value(np.mat(test_y[i]).T)

            for metrics_op in self.metrics_ops:
                metrics_op.forward()

            # self.logits.forward()
            # probs.append(self.logits.value.A1)

        # pred = np.array([1 if x >= 0.5 else 0 for x in probs])
        # accuracy = accuracy_score(test_y.flatten(), pred)
        # precision = precision_score(test_y.flatten(), pred)
        # recall = recall_score(test_y.flatten(), pred)
        # f1 = f1_score(test_y.flatten(), pred)

        # print("Sklearn Epoch: {:d}, Accuracy: {:.2f}%  Precision: {:.2f}% Recall: {:.2f}% F1Score: {:.2f}%".format(
        #     self.epoch + 1,  accuracy * 100, precision * 100, recall * 100, f1 * 100))

        metrics_str = 'Epoch [{}] '.format(self.epoch)
        for metrics_op in self.metrics_ops:
            metrics_str += metrics_op.value_str()
        print(metrics_str)

    def main_loop(self, train_x, train_y, test_x, test_y):
        '''
        训练（验证）的主循环
        '''
        for self.epoch in range(self.epoches):

            if self.eval_on_train and test_x is not None and test_y is not None:
                self.eval(test_x, test_y)

            # TODO improve the batch mechanism
            for i in range(len(train_x)):
                self.one_step(train_x[i], train_y[i])
                if i % self.batch_size == 0:
                    self.optimizer.update()
            print('Epoch [{}] train loss: {:.4f}'.format(
                self.epoch, float(self.loss_op.value)))

    def train(self, train_x, train_y, test_x=None, test_y=None):
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
