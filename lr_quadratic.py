# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 17:54:09 2020

@author: zhangjuefei
"""

import numpy as np
import matrixslow as ms
from sklearn.datasets import make_circles

# 获取同心圆状分布的数据，X的每行包含两个特征，y是1/0类别标签
X, y = make_circles(200, noise=0.1, factor=0.5)
y = y * 2 - 1  # 将标签转化为1/-1

# 是否使用二次项
use_quadratic = True

# 一次项，2维向量（2x1矩阵）
x1 = ms.core.Variable(dim=(2, 1), init=False, trainable=False)

# 标签
label = ms.core.Variable(dim=(1, 1), init=False, trainable=False)

# 偏置
b = ms.core.Variable(dim=(1, 1), init=True, trainable=True)

# 根据是否使用二次项区别处理
if use_quadratic:

    # 将一次项与自己的转置相乘，得到二次项2x2矩阵，再转成4维向量（4x1矩阵）
    x2 = ms.ops.Reshape(
            ms.ops.MatMul(x1, ms.ops.Reshape(x1, shape=(1, 2))),
            shape=(4, 1)
            )

    # 将一次和二次特征连接成6维向量（6x1矩阵）
    x = ms.ops.Concat(x1, x2)
    
    # 权值向量是6维（1x6矩阵）
    w = ms.core.Variable(dim=(1, 6), init=True, trainable=True)
    
else:
    
    # 特征向量就是一次项
    x = x1
    
    # 权值向量是2维（1x2矩阵）
    w = ms.core.Variable(dim=(1, 2), init=True, trainable=True)


# 线性部分
output = ms.ops.Add(ms.ops.MatMul(w, x), b)

# 预测概率
predict = ms.ops.Logistic(output)

# 损失函数
loss = ms.ops.loss.LogLoss(ms.ops.Multiply(label, output))

learning_rate = 0.001

optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)


batch_size = 8

for epoch in range(200):
    
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
        label.set_value(np.mat(y[i]))
        
        predict.forward()
        pred.append(predict.value[0, 0])
            
    pred = (np.array(pred) > 0.5).astype(np.int) * 2 - 1
    
    accuracy = (y == pred).astype(np.int).sum() / len(X)
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))