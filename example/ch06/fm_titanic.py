# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 20:38:10 2020

@author: chaos
"""

import sys
sys.path.append('../..')

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matrixslow as ms

# 读取数据，去掉无用列
data = pd.read_csv("../../data/titanic.csv").drop(["PassengerId", 
                  "Name", "Ticket", "Cabin"], axis=1)

# 构造编码类
le = LabelEncoder()
ohe = OneHotEncoder(sparse=False)

# 对类别型特征做One-Hot编码
Pclass = ohe.fit_transform(le.fit_transform(data["Pclass"].fillna(0)).reshape(-1, 1))
Sex = ohe.fit_transform(le.fit_transform(data["Sex"].fillna("")).reshape(-1, 1))
Embarked = ohe.fit_transform(le.fit_transform(data["Embarked"].fillna("")).reshape(-1, 1))

# 组合特征列
features = np.concatenate([Pclass,
                           Sex,
                           data[["Age"]].fillna(0),
                           data[["SibSp"]].fillna(0),
                           data[["Parch"]].fillna(0),
                           data[["Fare"]].fillna(0),
                           Embarked
                           ], axis=1)

# 标签
labels = data["Survived"].values * 2 - 1

# 特征维数
dimension = features.shape[1]

# 隐藏向量维度
k = 12

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
    for i in range(len(features)):
        
        x1.set_value(np.mat(features[i]).T)
        label.set_value(np.mat(labels[i]))
        
        optimizer.one_step()
        
        batch_count += 1
        if batch_count >= batch_size:
            
            optimizer.update()
            batch_count = 0
        

    pred = []
    for i in range(len(features)):
                
        x1.set_value(np.mat(features[i]).T)
        
        predict.forward()
        pred.append(predict.value[0, 0])
            
    pred = (np.array(pred) > 0.5).astype(np.int) * 2 - 1
    accuracy = (labels == pred).astype(np.int).sum() / len(features)
       
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))
