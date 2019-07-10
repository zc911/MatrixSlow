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

from core.graph import default_graph
from core.node import Variable
from ops.loss import LogLoss
from ops.ops import Add, Logistic, MatMul
from optimizer.optimizer import GradientDescent, AdaGrad, RMSProp, Adam, Momentum

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
                 weights[1][0] * x2 - bias[0][0]) / weights[2][0]
            ax.plot_surface(x1, x2, y)
        else:
            y = (-weights[0][0] * x1 - bias[0][0]) / weights[1][0]
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
    构建LR计算图模型
    '''
    x = Variable((1, feature_num), init=False, trainable=False)
    w = Variable((feature_num, 1), init=True, trainable=True)
    b = Variable((1, 1), init=True, trainable=True)

    logit = Add(MatMul(x, w), b)
    return x, logit, w, b


def train(train_x, train_y, test_x, test_y, epoches):

    x, logit, w, b = build_model(FEATURE_DIM)
    y = Variable((1, 1), init=False, trainable=False)
    # 对logit施加Logistic函数(sigmoid)
    y_hat = Logistic(logit)
    # 计算预测值和标签值的log loss，作为损失函数
    loss = LogLoss(y_hat, y)

    # 使用各种优化算法算法
    opt_type = 'Momentum'
    if opt_type == 'GradientDescent':
        optimizer = GradientDescent(default_graph, loss, 0.02, 16)
    elif opt_type == 'AdaGrad':
        optimizer = AdaGrad(default_graph, loss, 0.02, 0.9, 16)
    elif opt_type == 'RMSProp':
        optimizer = RMSProp(default_graph, loss, 0.02, 0.9, 16)
    elif opt_type == 'Momentum':
        optimizer = Momentum(default_graph, loss, 0.02, 0.9, 16)
    else:
        optimizer = Adam(default_graph, loss, 0.02, 0.9, 0.99, 16)

    for epoch in range(epoches):
        # 每个 epoch 开始时在测试集上评估模型正确率
        probs = []
        losses = []
        for i in range(len(test_x)):
            x.set_value(np.mat(test_x[i, :]))
            y.set_value(np.mat(test_y[i, 0]))

            # 前向传播计算概率
            y_hat.forward()
            probs.append(y_hat.value.A1)

            # 计算损失值
            loss.forward()
            losses.append(loss.value.A1)

        # 大于0.5的认为是正样本
        pred = np.array([1 if x >= 0.5 else 0 for x in probs])
        accuracy = accuracy_score(test_y.flatten(), pred)

        print("Epoch: {:d}，损失值：{:.3f}，正确率：{:.2f}%".format(
            epoch + 1, np.mean(losses), accuracy * 100))

        for i in range(len(train_x)):
            x.set_value(np.mat(train_x[i, :]))
            y.set_value(np.mat(train_y[i, 0]))
            optimizer.one_step()

    # 返回训练好的模型参数
    return w, b


FEATURE_DIM = 20
if __name__ == '__main__':
    # 随机构造训练数据
    train_x, train_y, test_x, test_y = random_gen_dateset(FEATURE_DIM, 1500)
    w, b = train(train_x, train_y, test_x, test_y, 8)
    plot_data(test_x, test_y, w.value, b.value)
