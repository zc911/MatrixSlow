# -*- coding: utf-8 -*-

import sys
from core.node import Variable
from ops.ops import Add, MatMul, Logistic
from ops.loss import LogLoss
from optimizer.optimizer import GradientDescent
from core.graph import default_graph
from util.test_data_util import get_data
import numpy as np

sys.path.append('.')


def build_model(feature_num):
    x = Variable((1, feature_num), init=False, trainable=False)
    w = Variable((feature_num, 1), init=True, trainable=True)
    b = Variable((1, 1), init=True, trainable=True)

    logit = Add(MatMul(x, w), b)
    return x, logit


def train(train_x, train_y, epoches):
    y = Variable((1, 1), init=False, trainable=False)

    x, logit = build_model(FEATURE_DIM)
    y_hat = Logistic(logit)

    losses = LogLoss(y_hat, y)
    optimizer = GradientDescent(default_graph, losses, 0.02, batch_size=8)
    for epoch in range(epoches):
        for i in range(len(train_x)):
            x.set_value(np.mat(train_x[i, :]))
            y.set_value(np.mat(train_y[i, 0]))
            optimizer.one_step()
            print(losses.value)


def eval():
    pass


FEATURE_DIM = 8
if __name__ == '__main__':
    # 构造训练数据
    train_x, train_y, test_x, test_y = get_data(
        2, number_of_features=FEATURE_DIM)
    train(train_x, train_y, 5)
