# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 15:58:54 CST 2019

@author: chenzhen
"""

from .trainer import Trainer


class SimpleTrainer(Trainer):

    def __init__(self, *args, **kargs):
        Trainer.__init__(self, *args, **kargs)

    def _variable_weights_init(self):
        '''
        不做统一的初始化操作，使用节点自身的初始化方法
        '''
        pass

    def _optimizer_update(self):
        self.optimizer.update()
