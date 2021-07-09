# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:13:19 2020

@author: chaos
"""

import sys
sys.path.append('../..')

import numpy as np
from sklearn.datasets import make_circles
import matrixslow as ms

X, y = make_circles(600, noise=0.1, factor=0.2)
y = y * 2 - 1

# 特征维数
dimension = 20

# 构造噪声特征
X = np.concatenate([X, np.random.normal(0.0, 0.5, (600, dimension-2))], axis=1)

# 隐藏向量维度
k = 2

# 一次项
x1 = ms.core.Variable(dim=(dimension, 1), init=False, trainable=False)

# 标签
label = ms.core.Variable(dim=(1, 1), init=False, trainable=False)

# 一次项权值向量
w = ms.core.Variable(dim=(1, dimension), init=True, trainable=True)

# 隐藏向量矩阵
H = ms.core.Variable(dim=(k, dimension), init=True, trainable=True)
HTH = ms.ops.MatMul(ms.ops.Transpose(H), H)

# 偏置
b = ms.core.Variable(dim=(1, 1), init=True, trainable=True)

# 线性部分
output = ms.ops.Add(
        ms.ops.MatMul(w, x1),   # 一次部分
        
        # 二次部分
        ms.ops.MatMul(ms.ops.Transpose(x1),
                      ms.ops.MatMul(HTH, x1)),
        b)

# 预测概率
predict = ms.ops.Logistic(output)

# 损失函数
loss = ms.ops.loss.LogLoss(ms.ops.Multiply(label, output))

learning_rate = 0.001
optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)


batch_size = 16

for epoch in range(50):
    
    batch_count = 0   
    for i in range(len(X)):
        
        x1.set_value(np.mat(X[i]).T)
        label.set_value(np.mat(y[i]))
        
        optimizer.one_step()
        
        batch_count += 1
        if batch_count >= batch_size:
            
            optimizer.update()
            batch_count = 0
        

    pred = []
    for i in range(len(X)):
                
        x1.set_value(np.mat(X[i]).T)
        
        predict.forward()
        pred.append(predict.value[0, 0])
            
    pred = (np.array(pred) > 0.5).astype(np.int) * 2 - 1
    accuracy = (y == pred).astype(np.int).sum() / len(X)
       
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))
