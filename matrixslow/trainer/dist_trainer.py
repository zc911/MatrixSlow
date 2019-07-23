# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 20:48:16 CST 2019

@author: chenzhen
"""
from . import trainer
from dist import ps


class SyncTrainerParameterServer(trainer.Trainer):
    def __init__(self, *args, **kargs):
        trainer.Trainer.__init__(self, *args, **kargs)
        self.ps_client = ps.ParameterServiceClient(ip='localhost', port=50051)

    def main_loop(self, train_x, train_y, test_x, test_y):
        '''
        训练（验证）的主循环
        '''
        for self.epoch in range(self.epoches):

            # TODO improve the batch mechanism
            for i in range(len(train_x)):
                self.one_step(train_x[i], train_y[i])
                if i % self.batch_size == 0:
                    acc_gradient = self.optimizer.acc_gradient
                    print(type(acc_gradient))
                    init
                    self.ps_client.push_gradients(acc_gradient)
                #     self.optimizer.update()
            print('Epoch [{}] train loss: {:.4f}'.format(
                self.epoch + 1, float(self.loss_op.value)))

            if self.eval_on_train and test_x is not None and test_y is not None:
                self.eval(test_x, test_y)
