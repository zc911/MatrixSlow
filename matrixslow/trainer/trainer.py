# -*- coding: utf-8 -*-

"""
Created on Wed Jul 10 15:19:42 CST 2019

@author: chenzhen
"""
import numpy as np


class Trainer(object):
    '''
    训练器
    '''

    def __init__(self, input_x, input_y, logits,
                 loss_op, optimizer,
                 epoches, batch_size=8,
                 eval_on_train=False, metrics_ops=None):
        self.input_x = input_x
        self.input_y = input_y
        self.logits = logits
        self.loss_op = loss_op
        self.optimizer = optimizer

        self.epoches = epoches
        self.epoch = 0
        self.batch_size = batch_size
        self.eval_on_train = eval_on_train
        self.metrics_ops = metrics_ops

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
        for metrics_op in self.metrics_ops:
            metrics_op.reset_value()

        for i in range(len(test_x)):
            self.input_x.set_value(np.mat(test_x[i]).T)
            self.input_y.set_value(np.mat(test_y[i]).T)

            for metrics_op in self.metrics_ops:
                metrics_op.forward()

        metrics_str = 'Epoch [{}] evaluation metrics '.format(self.epoch + 1)
        for metrics_op in self.metrics_ops:
            metrics_str += metrics_op.value_str()
        print(metrics_str)

    def main_loop(self, train_x, train_y, test_x, test_y):
        '''
        训练（验证）的主循环
        '''
        for self.epoch in range(self.epoches):

            # TODO improve the batch mechanism
            for i in range(len(train_x)):
                self.one_step(train_x[i], train_y[i])
                if i % self.batch_size == 0:
                    self.optimizer.update()
            print('Epoch [{}] train loss: {:.4f}'.format(
                self.epoch + 1, float(self.loss_op.value)))

            if self.eval_on_train and test_x is not None and test_y is not None:
                self.eval(test_x, test_y)

    def train(self, train_x, train_y, test_x=None, test_y=None):
        '''
        开始训练(验证)流程
        '''
        assert len(train_x) == len(train_y)
        if test_x is not None and test_y is not None:
            assert len(test_x) == len(test_y)

        # 传入数据，开始主循环
        self.main_loop(train_x, train_y, test_x, test_y)
