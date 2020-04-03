# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:13:26 2020

@author: zhangjuefei
"""

import numpy as np
from sklearn.datasets import fetch_openml
import matrixslow as ms

# 加载MNIST数据集，取一部分样本并归一化
# X, _ = fetch_openml('mnist_784', version=1, return_X_y=True)


# 输入图像尺寸
img_shape = (28, 28)

# 输入图像
x = ms.core.Variable(img_shape, init=False, trainable=False)

# 标签
real_or_fake = ms.core.Variable(dim=(2, 1), init=False, trainable=False)


# 生成网络
rand = ms.core.Variable(dim=(100, 1), init=False, trainable=False)
tmp = ms.layer.fc(rand, 100, 200, "ReLU")
tmp = ms.layer.fc(tmp, 200, 784, "ReLU")
fake = ms.ops.Reshape(tmp, shape=img_shape)

g_graph = ms.core.Graph()
g_graph.nodes = ms.default_graph.nodes.copy()
ms.default_graph.nodes.clear()


# 判别CNN
d_input = ms.ops.Welding()
conv = ms.layer.conv([d_input], img_shape, 3, (3, 3), "ReLU")
pooling = ms.layer.pooling(conv, (3, 3), (2, 2))
fc = ms.layer.fc(ms.ops.Concat(*pooling), 588, 100, "ReLU")
output = ms.layer.fc(fc, 100, 2, "None")
predict = ms.ops.SoftMax(output)

d_graph = ms.core.Graph()
d_graph.nodes = ms.default_graph.nodes.copy()

# 交叉熵损失
loss = ms.ops.loss.CrossEntropyWithSoftMax(output, real_or_fake)

# 学习率
learning_rate = 0.004
g_optimizer = ms.optimizer.Adam(g_graph, loss, learning_rate)
d_optimizer = ms.optimizer.Adam(d_graph, loss, learning_rate)


# 训练
batch_size = 32

for epoch in range(30):
    
    batch_count = 0
    
    for i in range(len(X)):
        
        # 训练判别网络
        feature = np.mat(X[i]).reshape(img_shape)
        x.set_value(feature)
        
        d_input.weld(x)
        real_or_fake.set_value(np.mat([1, 0]).T)
        

        d_optimizer.one_step()
        
        
        rand.set_value(np.mat(np.random.multivariate_normal([0]*100, np.eye(100) * 0.4, 1)).T)
        d_input.weld(fake)
        real_or_fake.set_value(np.mat([0, 1]).T)
        
        d_optimizer.one_step()

        
        # 训练生成网络
        rand.set_value(np.mat(np.random.multivariate_normal([0]*100, np.eye(100) * 0.4, 1)).T)
        d_input.weld(fake)
        real_or_fake.set_value(np.mat([1, 0]).T)
        
        g_optimizer.one_step()
        
        
        batch_count += 1
        if batch_count >= batch_size:
            
            print("epoch: {:d}, iteration: {:d}, loss: {:.3f}".format(epoch + 1, i + 1, loss.value[0, 0]))

            d_optimizer.update()
            g_optimizer.update()
            batch_count = 0
