#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: chenzhen
@Date: 2020-04-10 17:04:46
@LastEditTime: 2020-04-24 15:45:41
@LastEditors: chenzhen
@Description:
'''

import sys
sys.path.append('../../')

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
import matrixslow as ms
from matrixslow.trainer import SimpleTrainer


# 输入图像尺寸
img_shape = (28, 28)

# 加载MNIST数据集，取一部分样本并归一化
test_data, test_label = fetch_openml(
    'mnist_784', version=1, return_X_y=True, cache=True)
test_data, test_label = test_data[1000:2000] / \
    255, test_label.astype(np.int)[1000:2000]
test_data = np.reshape(np.array(test_data), (1000, *img_shape))


saver = ms.trainer.Saver('./epoches10')

saver.load(model_file_name='my_model.json', weights_file_name='my_weights.npz')

# 根据训练时定义的节点名称，从计算图中把输入输出节点查询出来
# 如果训练时未定义，节点名称自动生成，需要从模型文件中人为识别出来
x = ms.get_node_from_graph('img_input')
pred = ms.get_node_from_graph('softmax_output')

for index in range(len(test_data)):
    # 把预测数据赋值给输入节点
    x.set_value(np.mat(test_data[index]).T)
    # 执行前向传播，计算输出节点的值，即模型预测概率
    pred.forward()
    gt = test_label[index]
    print('model predict {} and ground truth: {}'.format(
        np.argmax(pred.value), gt))
