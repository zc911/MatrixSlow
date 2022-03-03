# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:34:01 2020

@author: zhangjuefei
"""

import sys
sys.path.append('../..')

import matrixslow as ms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 读取图像，归一化
pic = matplotlib.image.imread('../../data/lena.jpg') / 255

# 图像尺寸
w, h = pic.shape

# Sobel滤波器
sobel = ms.core.Variable(dim=(3, 3), init=False, trainable=False)
sobel.set_value(np.mat([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))

# 输入图像
img = ms.core.Variable(dim=(w, h), init=False, trainable=False)
img.set_value(np.mat(pic))

# Sobel滤波器输出
sobel_output = ms.ops.Convolve(img, sobel)

# 输出图像
sobel_output.forward()
plt.imshow(sobel_output.value, cmap="gray")

# 可训练滤波器
filter_train = ms.core.Variable(dim=(3, 3), init=True, trainable=True)
filter_output = ms.ops.Convolve(img, filter_train)

# 常数矩阵：-1
minus = ms.core.Variable(dim=(w, h), init=False, trainable=False)
minus.set_value(np.mat(-np.ones((w, h))))

# 常数（矩阵）：图像总像素数的倒数
n = ms.core.Variable((1, 1), init=False, trainable=False)
n.set_value(np.mat(1.0 / (w * h)))

# 损失值，均方误差
error = ms.ops.Add(sobel_output, ms.ops.Multiply(filter_output, minus))
square_error = ms.ops.MatMul(
                    ms.ops.Reshape(error, shape=(1, w * h)), 
                    ms.ops.Reshape(error, shape=(w * h, 1))
                )

mse = ms.ops.MatMul(square_error, n)

# 优化器
optimizer = ms.optimizer.Adam(ms.core.default_graph, mse, 0.01)

# 训练
for i in range(1000):
    
    optimizer.one_step()
    optimizer.update()               
    mse.forward()
    print("iteration:{:d},loss:{:.10f}".format(i, mse.value[0, 0]))


# 被训练完成的滤波器
filter_train.forward()
print(filter_train.value)

# 用被训练的滤波器滤波图像
filter_output.forward()
plt.imshow(filter_output.value, cmap="gray")
