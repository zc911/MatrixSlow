# -*- coding: utf-8 -*-
"""
Created on Wed July  9 15:13:01 2019

@author: chenzhen
"""
import random
import sys

import matplotlib
import numpy as np
from sklearn.metrics import accuracy_score

from core import Variable
from core.graph import default_graph
from ops import Add, Logistic, MatMul, ReLU, SoftMax
from ops.loss import CrossEntropyWithSoftMax, LogLoss
from ops.metrics import Metrics
from optimizer import *
from trainer import Trainer
from util import ClassMining

matplotlib.use('TkAgg')
sys.path.append('.')


def plot_data(data_x, data_y, weights=None, bias=None):
    '''
    绘制数据节点和线性模型，只绘制2维或3维
    如果特征维度>3,默认使用前3个特征绘制
    '''
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    assert len(data_x) == len(data_y)
    data_dim = data_x.shape[1]
    plot_3d = False if data_dim < 3 else True

    xcord1 = []
    ycord1 = []
    zcord1 = []
    xcord2 = []
    ycord2 = []
    zcord2 = []
    for i in range(data_x.shape[0]):
        if int(data_y[i]) == 1:
            xcord1.append(data_x[i, 0])
            ycord1.append(data_x[i, 1])
            if plot_3d:
                zcord1.append(data_x[i, 2])
        else:
            xcord2.append(data_x[i, 0])
            ycord2.append(data_x[i, 1])
            if plot_3d:
                zcord2.append(data_x[i, 2])
    fig = plt.figure()
    if plot_3d:
        ax = Axes3D(fig)
        ax.scatter(xcord1, ycord1, zcord1, s=30, c='red', marker='s')
        ax.scatter(xcord2, ycord2, zcord2, s=30, c='green')
    else:
        ax = fig.add_subplot(111)
        ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
        ax.scatter(xcord2, ycord2, s=30, c='green')

    if weights is not None and bias is not None:
        x1 = np.arange(-1.0, 1.0, 0.1)
        if plot_3d:
            x2 = np.arange(-1.0, 1.0, 0.1)
            x1, x2 = np.meshgrid(x1, x2)
        weights = np.array(weights)
        bias = np.array(bias)
        if plot_3d:
            y = (-weights[0][0] * x1 -
                 weights[0][1] * x2 - bias[0][0]) / weights[0][2]
            ax.plot_surface(x1, x2, y)
        else:
            y = (-weights[0][0] * x1 - bias[0][0]) / weights[0][1]
            ax.plot(x1, y)
    plt.show()


def random_gen_dateset(feature_num, sample_num, test_radio=0.3, seed=41):
    '''
    生成二分类样本
    '''
    random.seed(seed)
    rand_bias = np.mat(np.random.uniform(-0.1, 0.1, (sample_num, 1)))
    rand_weights = np.mat(np.random.uniform(-1, 1, (feature_num, 1)))
    data_x = np.mat(np.random.uniform(-1, 1, (sample_num, feature_num)))
    data_y = (data_x * rand_weights) + rand_bias
    data_y = np.where(data_y > 0, 1, 0)
    train_size = int(sample_num * (1 - test_radio))

    return (data_x[:train_size, :],
            data_y[:train_size, :],
            data_x[train_size:, :],
            data_y[train_size:, :])


def build_model(feature_num):
    '''
    构建DNN计算图网络
    '''
    x = Variable((feature_num, 1), init=False, trainable=False)
    w1 = Variable((HIDDEN1_SIZE, feature_num), init=True, trainable=True)
    b1 = Variable((HIDDEN1_SIZE, 1), init=True, trainable=True)
    w2 = Variable((HIDDEN2_SIZE, HIDDEN1_SIZE), init=True, trainable=True)
    b2 = Variable((HIDDEN2_SIZE, 1), init=True, trainable=True)
    w3 = Variable((CLASSES, HIDDEN2_SIZE), init=True, trainable=True)
    b3 = Variable((CLASSES, 1), init=True, trainable=True)

    hidden1 = ReLU(Add(MatMul(w1, x), b1))
    hidden2 = ReLU(Add(MatMul(w2, hidden1), b2))
    logit = Add(MatMul(w3, hidden2), b3)

    return x, logit, w1, b1


def build_metrics(logits, y, metrics_names=None):
    metrics_ops = []
    for m_name in metrics_names:
        metrics_ops.append(ClassMining.get_instance_by_subclass_name(
            Metrics, m_name)(logits, y))

    return metrics_ops


def train(train_x, train_y, test_x, test_y, epoches, batch_size):

    x, logits, w, b = build_model(FEATURE_DIM)

    y = Variable((CLASSES, 1), init=False, trainable=False)
    loss_op = CrossEntropyWithSoftMax(logits, y)
    optimizer_op = optimizer.Momentum(default_graph, loss_op)
    trainer = Trainer(x, y, logits, loss_op, optimizer_op,
                      epoches=epoches, batch_size=batch_size,
                      eval_on_train=True,
                      metrics_ops=build_metrics(
                          logits, y, ['Accuracy', 'Recall', 'F1Score', 'Precision']))
    trainer.train(train_x, train_y, test_x, test_y)

    return w, b


SAMPLE_NUM = 1000
FEATURE_DIM = 784
TOTAL_EPOCHES = 5
BATCH_SIZE = 32
HIDDEN1_SIZE = 12
HIDDEN2_SIZE = 8
CLASSES = 10
if __name__ == '__main__':
    train_x, train_y, test_x, test_y = util.mnist('../MNIST/dataset')
    w, b = train(train_x, train_y, test_x, test_y, TOTAL_EPOCHES, BATCH_SIZE)
    # plot_data(test_x, test_y, w.value, b.value)
