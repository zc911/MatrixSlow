# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:34:46 CST 2019

@author: chenzhen
"""

import numpy as np

from core import Node
import matrixslow as ms


class Metrics(Node):
    '''
    评估指标算子抽象基类
    '''

    def __init__(self, *parents, **kargs):
        # 默认情况下，metrics节点不需要保存
        kargs['need_save'] = kargs.get('need_save', False)
        Node.__init__(self, *parents, **kargs)

    @staticmethod
    def prob_to_label(prob):
        if prob.shape[0] > 1:
            # 如果是多分类，预测值为概率最大的标签
            labels = np.argmax(prob, axis=0)
        else:
            # 否则以0.5作为thresholds
            labels = np.where(prob < 0.5, 0, 1)
        return labels

    def get_jacobi(self):
        # 对于评估指标算子，计算雅各比无意义
        raise NotImplementedError()

    def value_str(self):
        return "{}: {:.4f} ".format(self.__class__.__name__, self.value)


class Accuracy(Metrics):
    '''
    Accuracy算子
    '''

    def __init__(self, *parents, **kargs):
        Metrics.__init__(self, *parents, **kargs)
        self.correct_num = 0
        self.total_num = 0

    def compute(self):
        '''
        计算Accrucy: (TP + TN) / TOTAL
        这里假设第一个父节点是预测值（概率），第二个父节点是标签
        '''

        pred = Metrics.prob_to_label(self.parents[0].value)
        gt = Metrics.prob_to_label(self.parents[1].value)
        assert len(pred) == len(gt)

        self.correct_num += np.sum(pred == gt)
        self.total_num += len(pred)
        self.value = 0
        if self.total_num != 0:
            self.value = float(self.correct_num) / self.total_num


class Precision(Metrics):
    '''
    Precision算子
    '''

    def __init__(self, *parents, **kargs):
        Metrics.__init__(self, *parents, **kargs)
        self.true_pos_num = 0
        self.pred_pos_num = 0

    def compute(self):
        '''
        计算Precision： TP / (TP + FP)
        '''
        assert self.parents[0].value.shape[1] == 1

        pred = Metrics.prob_to_label(self.parents[0].value)
        gt = self.parents[1].value
        self.pred_pos_num += np.sum(pred)
        self.true_pos_num += np.multiply(pred, gt).sum()
        self.value = 0
        if self.pred_pos_num != 0:
            self.value = float(self.true_pos_num) / self.pred_pos_num


class Recall(Metrics):
    '''
    Recall算法
    '''

    def __init__(self, *parents, **kargs):
        Metrics.__init__(self, *parents, **kargs)
        self.gt_pos_num = 0
        self.true_pos_num = 0

    def compute(self):
        '''
        计算Recall： TP / (TP + FN)
        '''
        assert self.parents[0].value.shape[1] == 1

        pred = Metrics.prob_to_label(self.parents[0].value)
        gt = self.parents[1].value
        self.gt_pos_num += np.sum(gt)
        self.true_pos_num += np.multiply(pred, gt).sum()
        self.value = 0
        if self.gt_pos_num != 0:
            self.value = float(self.true_pos_num) / self.gt_pos_num


class F1Score(Metrics):
    '''
    F1 Score算子

    '''

    def __init__(self, *parents, **kargs):
        '''
        F1Score算子
        '''
        Metrics.__init__(self, *parents, **kargs)
        self.true_pos_num = 0
        self.pred_pos_num = 0
        self.gt_pos_num = 0

    def compute(self):
        '''
        计算f1-score: (2 * pre * recall) / (pre + recall)
        '''

        assert self.parents[0].value.shape[1] == 1

        pred = Metrics.prob_to_label(self.parents[0].value)
        gt = self.parents[1].value
        self.gt_pos_num += np.sum(gt)
        self.pred_pos_num += np.sum(pred)
        self.true_pos_num += np.multiply(pred, gt).sum()
        self.value = 0
        pre_score = 0
        recall_score = 0

        if self.pred_pos_num != 0:
            pre_score = float(self.true_pos_num) / self.pred_pos_num

        if self.gt_pos_num != 0:
            recall_score = float(self.true_pos_num) / self.gt_pos_num

        self.value = 0
        if pre_score + recall_score != 0:
            self.value = 2 * \
                np.multiply(pre_score, recall_score) / \
                (pre_score + recall_score)
