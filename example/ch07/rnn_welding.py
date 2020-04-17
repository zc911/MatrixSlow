# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 11:02:28 2020

@author: chaos
"""

import numpy as np
import matrixslow as ms
from scipy import signal


# 构造正弦波和方波两类样本的函数 
# 构造正弦波和方波两类样本的函数
def get_sequence_data(dimension=10, length=10,
                      number_of_examples=1000, train_set_ratio=0.7, seed=42):
    """
    生成两类序列数据。
    """
    xx = []

    # 正弦波
    xx.append(np.sin(np.arange(0, 10, 10 / length)).reshape(-1, 1))

    # 方波
    xx.append(np.array(signal.square(np.arange(0, 10, 10 / length))).reshape(-1, 1))

    data = []
    for i in range(2):
        x = xx[i]
        for j in range(number_of_examples // 2):
            sequence = x + np.random.normal(0, 0.6, (len(x), dimension))  # 加入噪声
            label = np.array([int(i == k) for k in range(2)])
            data.append(np.c_[sequence.reshape(1, -1), label.reshape(1, -1)])

    # 把各个类别的样本合在一起
    data = np.concatenate(data, axis=0)

    # 随机打乱样本顺序
    np.random.shuffle(data)

    # 计算训练样本数量
    train_set_size = int(number_of_examples * train_set_ratio)  # 训练集样本数量

    # 将训练集和测试集、特征和标签分开
    return (data[:train_set_size, :-2].reshape(-1, length, dimension),
            data[:train_set_size, -2:],
            data[train_set_size:, :-2].reshape(-1, length, dimension),
            data[train_set_size:, -2:])


# 构造RNN
seq_len = 96  # 序列长度
dimension = 16  # 输入维度
status_dimension = 12  # 状态维度

signal_train, label_train, signal_test, label_test = get_sequence_data(length=seq_len, dimension=dimension)

# 输入向量节点
inputs = [ms.core.Variable(dim=(dimension, 1), init=False, trainable=False) for i in range(seq_len)]
 
# 输入权值矩阵
U = ms.core.Variable(dim=(status_dimension, dimension), init=True, trainable=True)

# 状态权值矩阵
W = ms.core.Variable(dim=(status_dimension, status_dimension), init=True, trainable=True)

# 偏置向量
b = ms.core.Variable(dim=(status_dimension, 1), init=True, trainable=True)

# 保存各个时刻内部状态变量的数组
hiddens = []

last_step = None
for iv in inputs:
    h = ms.ops.Add(ms.ops.MatMul(U, iv), b)

    if last_step is not None:
        h = ms.ops.Add(ms.ops.MatMul(W, last_step), h)

    h = ms.ops.ReLU(h)

    last_step = h
    hiddens.append(last_step)


# 焊接点，暂时不连接父节点
welding_point = ms.ops.Welding()


# 全连接网络
fc1 = ms.layer.fc(welding_point, status_dimension, 40, "ReLU")
fc2 = ms.layer.fc(fc1, 40, 10, "ReLU")
output = ms.layer.fc(fc2, 10, 2, "None")

# 概率
predict = ms.ops.Logistic(output)

# 训练标签
label = ms.core.Variable((2, 1), trainable=False)

# 交叉熵损失
loss = ms.ops.CrossEntropyWithSoftMax(output, label)


# 训练
learning_rate = 0.005
optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)

batch_size = 16

for epoch in range(10):
    
    batch_count = 0   
    for i, s in enumerate(signal_train):
        
        # 取一个变长序列
        start = np.random.randint(len(s) // 3)
        end = np.random.randint(len(s) // 3 + 30, len(s))
        s = s[start: end]
        
        # 将变长的输入序列赋给RNN的各输入向量节点
        for j in range(len(s)):
            inputs[j].set_value(np.mat(s[j]).T)    
        
        # 将临时的最后一个时刻与全连接网络焊接
        welding_point.weld(hiddens[j])
        
        label.set_value(np.mat(label_train[i, :]).T)
        
        optimizer.one_step()
        
        batch_count += 1
        if batch_count >= batch_size:
            
            print("epoch: {:d}, iteration: {:d}, loss: {:.3f}".format(epoch + 1, i + 1, loss.value[0, 0]))

            
            optimizer.update()
            batch_count = 0
        

    pred = []
    for i, s in enumerate(signal_test):
        
        start = np.random.randint(len(s) // 3)
        end = np.random.randint(len(s) // 3 + 30, len(s))
        s = s[start: end]
        
        for j in range(len(s)):
            inputs[j].set_value(np.mat(s[j]).T)    
        
        welding_point.weld(hiddens[j])

        predict.forward()
        pred.append(predict.value.A.ravel())
            
    pred = np.array(pred).argmax(axis=1)
    true = label_test.argmax(axis=1)
    
    accuracy = (true == pred).astype(np.int).sum() / len(signal_test)
    print("epoch: {:d}, accuracy: {:.5f}".format(epoch + 1, accuracy))