# -*- coding: utf-8 -*-
import numpy as np


def get_data(number_of_classes=2, seed=42, number_of_features=5, number_of_examples=1000, train_set_ratio=0.7):
    np.random.seed(seed)

    # 对每一类别生成样本
    data = []
    for i in range(number_of_classes):
        h = np.mat(np.random.random(
            (number_of_features, number_of_features))) * 0.2
        features = np.random.multivariate_normal(
            mean=np.random.random(number_of_features),
            # 随机生成一个对称矩阵作为协方差矩阵，有可能不正定
            cov=h.T * h + 0.03 * np.mat(np.eye(number_of_features)),
            check_valid="raise",  # 万一不正定了，抛异常
            size=number_of_examples  # 样本数
        )

        labels = np.array(
            [[int(i == j) for j in range(number_of_classes)]] * number_of_examples)
        data.append(np.c_[features, labels])

    # 把各个类别的样本合在一起
    data = np.concatenate(data, axis=0)

    # 随机打乱样本顺序
    np.random.shuffle(data)

    # 计算训练样本数量
    train_set_size = int(number_of_examples * train_set_ratio)  # 训练集样本数量

    # 将训练集和测试集、特征和标签分开
    return (data[:train_set_size, :-number_of_classes],
            data[:train_set_size, -number_of_classes:],
            data[train_set_size:, :-number_of_classes],
            data[train_set_size:, -number_of_classes:])
