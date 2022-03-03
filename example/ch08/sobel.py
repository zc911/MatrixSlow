# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 10:07:59 2020

@author: chaos
"""

import sys
sys.path.append('../..')

import matrixslow as ms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 读取图像，归一化
pic = matplotlib.image.imread('../../data/mondrian.jpg') / 255

# 图像尺寸
w, h = pic.shape

# 纵向Sobel滤波器
sobel_v = ms.core.Variable(dim=(3, 3), init=False, trainable=False)
sobel_v.set_value(np.mat([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))

# 横向Sobel滤波器
sobel_h = ms.core.Variable(dim=(3, 3), init=False, trainable=False)
sobel_h.set_value(sobel_v.value.T)

# 输入图像
img = ms.core.Variable(dim=(w, h), init=False, trainable=False)
img.set_value(np.mat(pic))

# Sobel滤波器输出
sobel_v_output = ms.ops.Convolve(img, sobel_v)
sobel_h_output = ms.ops.Convolve(img, sobel_h)

# 两个Sobel滤波器的输出平方和
square_output = ms.ops.Add(
            ms.ops.Multiply(sobel_v_output, sobel_v_output),
            ms.ops.Multiply(sobel_h_output, sobel_h_output)
        )

# 前向传播
square_output.forward()

# 输出图像
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(221)
ax.axis("off")
ax.imshow(img.value, cmap="gray")

ax = fig.add_subplot(222)
ax.axis("off")
ax.imshow(square_output.value, cmap="gray")

ax = fig.add_subplot(223)
ax.axis("off")
ax.imshow(sobel_v_output.value, cmap="gray")

ax = fig.add_subplot(224)
ax.axis("off")
ax.imshow(sobel_h_output.value, cmap="gray")

plt.show()
