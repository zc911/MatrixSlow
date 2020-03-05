# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 22:54:59 2020

@author: chaos
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
import matrixslow as ms

# 加载MNIST数据集，只取5000个样本
# X, _ = fetch_openml('mnist_784', version=1, return_X_y=True)
# X = X[:5000] / 255


# 构造计算图：输入向量，是一个784x1矩阵，不需要初始化，不参与训练
x = ms.core.Variable(dim=(784, 1), init=False, trainable=False)


# hidden_1 = ms.layer.fc(x, 784, 100, "ReLU")


encoding = ms.layer.fc(x, 784, 20, "ReLU")


# hidden_2 = ms.layer.fc(encoding, 20, 100, "ReLU")


decoding = ms.layer.fc(encoding, 20, 784, "ReLU")

minus_one = ms.core.Variable(dim=(1, 1), init=False, trainable=False)
minus_one.set_value(np.mat([[-1]]))

N = ms.core.Variable(dim=(1, 1), init=False, trainable=False)
N.set_value(np.mat([[1/784]]))

minus_decoding = ms.ops.ScalarMultiply(minus_one, decoding)

diff = ms.ops.Add(decoding, x)

mean_square_diff = ms.ops.MatMul(
            N,
            ms.ops.MatMul(
                    ms.ops.Reshape(diff, shape=(1, 784)),
                    diff
            )
        )


# 学习率
learning_rate = 0.005

# 构造Adam优化器
optimizer = ms.optimizer.Adam(ms.default_graph, mean_square_diff, learning_rate)

# 批大小为64
batch_size = 16

# 训练执行10个epoch
for epoch in range(30):
    
    # 批计数器清零
    batch_count = 0
    
    # 遍历训练集中的样本
    for i in range(len(X)):
        
        # 取第i个样本，构造784x1矩阵对象
        feature = np.mat(X[i]).T
        
        # 将特征赋给x节点，将标签赋给one_hot节点
        x.set_value(feature)
        
        # 调用优化器的one_step方法，执行一次前向传播和反向传播
        optimizer.one_step()
        
        # 批计数器加1
        batch_count += 1
        
        # 若批计数器大于等于批大小，则执行一次梯度下降更新，并清零计数器
        if batch_count >= batch_size:
            
            # 打印当前epoch数，迭代数与损失值
            print("epoch: {:d}, iteration: {:d}, mean_square_diff: {:.3f}".format(epoch + 1, i + 1, mean_square_diff.value[0, 0]))

            # 优化器执行梯度下降更新
            optimizer.update()
            batch_count = 0