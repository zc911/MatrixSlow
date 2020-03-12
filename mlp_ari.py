# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:14:29 2020

@author: chaos
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.io import arff
from scipy import signal

import matrixslow as ms

path_train = "data/ArticularyWordRecognition/ArticularyWordRecognition_TRAIN.arff"
path_test = "data/ArticularyWordRecognition/ArticularyWordRecognition_TEST.arff"


# 读取arff格式数据
train, test = arff.loadarff(path_train), arff.loadarff(path_test)
train, test = pd.DataFrame(train[0]), pd.DataFrame(test[0])

# 整理数据格式，结果为316x50x28数组：316个样本，每个样本有50个时刻，每个时刻是28个信号
signal_train = np.array([np.array([list(channel) for channel in sample]).T for sample in train["relationalAtt"]])
signal_test = np.array([np.array([list(channel) for channel in sample]).T for sample in test["relationalAtt"]])
signal_train = signal_train.reshape((275, 1, -1)).squeeze()
signal_test = signal_test.reshape((300, 1, -1)).squeeze()

# 标签，One-Hot编码
le = LabelEncoder()
ohe = OneHotEncoder(sparse=False)
label_train = ohe.fit_transform(le.fit_transform(train["classAttribute"]).reshape(-1, 1))
label_test = ohe.fit_transform(le.fit_transform(test["classAttribute"]).reshape(-1, 1))


# 构造计算图：输入向量，是一个784x1矩阵，不需要初始化，不参与训练
x = ms.core.Variable(dim=(1296, 1), init=False, trainable=False)

# One-Hot类别标签，是10x1矩阵
one_hot = ms.core.Variable(dim=(25, 1), init=False, trainable=False)

# 输入层，100个神经元，激活函数为ReLU
hidden_1 = ms.layer.fc(x, 1296, 100, "ReLU")


# 输出层，10个神经元，无激活函数
output = ms.layer.fc(hidden_1, 100, 25, None)

# 概率输出
predict = ms.ops.SoftMax(output)

# 交叉熵损失
loss = ms.ops.loss.CrossEntropyWithSoftMax(output, one_hot)

# 学习率
learning_rate = 0.001

# 构造Adam优化器
optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)

# 批大小为64
batch_size = 64

# 训练执行10个epoch
for epoch in range(30):
    
    # 批计数器清零
    batch_count = 0
    
    # 遍历训练集中的样本
    for i, s in enumerate(signal_train):
        
        # 取第i个样本，构造784x1矩阵对象
        feature = np.mat(s).T
        
        # 取第i个样本的One-Hot标签，10x1矩阵
        label = np.mat(label_train[i]).T
        
        # 将特征赋给x节点，将标签赋给one_hot节点
        x.set_value(feature)
        one_hot.set_value(label)
        
        # 调用优化器的one_step方法，执行一次前向传播和反向传播
        optimizer.one_step()
        
        # 批计数器加1
        batch_count += 1
        
        # 若批计数器大于等于批大小，则执行一次梯度下降更新，并清零计数器
        if batch_count >= batch_size:
            
            # 打印当前epoch数，迭代数与损失值
            print("epoch: {:d}, iteration: {:d}, loss: {:.3f}".format(epoch + 1, i + 1, loss.value[0, 0]))

            # 优化器执行梯度下降更新
            optimizer.update()
            batch_count = 0
        

    # 每个epoch结束后评价模型的正确率
    pred = []
    
    # 遍历训练集，计算当前模型对每个样本的预测值
    for i, s in enumerate(signal_test):
                
        feature = np.mat(s).T
        x.set_value(feature)
        
        # 在模型的predict节点上执行前向传播
        predict.forward()
        pred.append(predict.value.A.ravel())  # 模型的预测结果：10个概率值
            
    pred = np.array(pred).argmax(axis=1)  # 取最大概率对应的类别为预测类别
    true = np.argmax(label_test, axis=1)
    
    # 判断预测结果与样本标签相同的数量与训练集总数量之比，即模型预测的正确率
    accuracy = (true == pred).astype(np.int).sum() / len(signal_test)
       
    # 打印当前epoch数和模型在训练集上的正确率
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))