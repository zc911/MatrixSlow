# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:58:19 2020

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

# 嵌入向量维度
k = 2

# 一次项
x = ms.core.Variable(dim=(dimension, 1), init=False, trainable=False)

# 三个类别类特征的三套One-Hot
x_Pclass = ms.core.Variable(dim=(Pclass.shape[1], 1), init=False, trainable=False)
x_Sex = ms.core.Variable(dim=(Sex.shape[1], 1), init=False, trainable=False)
x_Embarked = ms.core.Variable(dim=(Embarked.shape[1], 1), init=False, trainable=False)


# 标签
label = ms.core.Variable(dim=(1, 1), init=False, trainable=False)

# 一次项权值向量
w = ms.core.Variable(dim=(1, dimension), init=True, trainable=True)

# 类别类特征的嵌入矩阵
E_Pclass = ms.core.Variable(dim=(k, Pclass.shape[1]), init=True, trainable=True)
E_Sex = ms.core.Variable(dim=(k, Sex.shape[1]), init=True, trainable=True)
E_Embarked = ms.core.Variable(dim=(k, Embarked.shape[1]), init=True, trainable=True)

# 偏置
b = ms.core.Variable(dim=(1, 1), init=True, trainable=True)


# 三个嵌入向量
embedding_Pclass = ms.ops.MatMul(E_Pclass, x_Pclass)
embedding_Sex = ms.ops.MatMul(E_Sex, x_Sex)
embedding_Embarked = ms.ops.MatMul(E_Embarked, x_Embarked)

# 将三个嵌入向量连接在一起
embedding = ms.ops.Concat(
        embedding_Pclass,
        embedding_Sex,
        embedding_Embarked
        )


# FM部分
fm = ms.ops.Add(ms.ops.MatMul(w, x),   # 一次部分                
                # 二次部分
                ms.ops.MatMul(ms.ops.Transpose(embedding), embedding)
                )



# Deep部分，第一隐藏层
hidden_1 = ms.layer.fc(embedding, 3 * k, 8, "ReLU")

# 第二隐藏层
hidden_2 = ms.layer.fc(hidden_1, 8, 4, "ReLU")

# 输出层
deep = ms.layer.fc(hidden_2, 4, 1, None)

# 输出
output = ms.ops.Add(fm, deep, b)

# 预测概率
predict = ms.ops.Logistic(output)

# 损失函数
loss = ms.ops.loss.LogLoss(ms.ops.Multiply(label, output))

learning_rate = 0.005
optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)


batch_size = 16

for epoch in range(50):
    
    batch_count = 0   
    for i in range(len(features)):
        
        x.set_value(np.mat(features[i]).T)
        
        # 从特征中选择各段One-Hot编码
        x_Pclass.set_value(np.mat(features[i, :3]).T)
        x_Sex.set_value(np.mat(features[i, 3:5]).T)
        x_Embarked.set_value(np.mat(features[i, 9:]).T)
        
        label.set_value(np.mat(labels[i]))
        
        optimizer.one_step()
        
        batch_count += 1
        if batch_count >= batch_size:
            
            optimizer.update()
            batch_count = 0
        

    pred = []
    for i in range(len(features)):
                
        x.set_value(np.mat(features[i]).T)
        x_Pclass.set_value(np.mat(features[i, :3]).T)
        x_Sex.set_value(np.mat(features[i, 3:5]).T)
        x_Embarked.set_value(np.mat(features[i, 9:]).T)
        
        predict.forward()
        pred.append(predict.value[0, 0])
            
    pred = (np.array(pred) > 0.5).astype(np.int) * 2 - 1
    accuracy = (labels == pred).astype(np.int).sum() / len(features)
       
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))
