import sys
sys.path.append('../..')
import numpy as np
import matrixslow as ms

# 构造训练集
male_heights = np.random.normal(171, 6, 500)
female_heights = np.random.normal(158, 5, 500)

male_weights = np.random.normal(70, 10, 500)
female_weights = np.random.normal(57, 8, 500)

male_bfrs = np.random.normal(16, 2, 500)
female_bfrs = np.random.normal(22, 2, 500)

male_labels = [1] * 500
female_labels = [-1] * 500

train_set = np.array([np.concatenate((male_heights, female_heights)),
                      np.concatenate((male_weights, female_weights)),
                      np.concatenate((male_bfrs, female_bfrs)),
                      np.concatenate((male_labels, female_labels))]).T

# 随机打乱样本顺序
np.random.shuffle(train_set)

# 批大小
batch_size = 10

# batch_size x 3矩阵，每行保存一个样本，整个节点保存一个mini batch的样本
X = ms.core.Variable(dim=(batch_size, 3), init=False, trainable=False)

# 保存一个mini batch的样本的类别标签
label = ms.core.Variable(dim=(batch_size, 1), init=False, trainable=False)

# 权值向量，3x1矩阵
w = ms.core.Variable(dim=(3, 1), init=True, trainable=True)

# 阈值
b = ms.core.Variable(dim=(1, 1), init=True, trainable=True)

# 全1向量，维数是batch_size，不可训练
ones = ms.core.Variable(dim=(batch_size, 1), init=False, trainable=False)
ones.set_value(np.mat(np.ones(batch_size)).T)

# 用阈值（标量）乘以全1向量
bias = ms.ops.ScalarMultiply(b, ones)

# 对一个mini batch的样本计算输出
output = ms.ops.Add(ms.ops.MatMul(X, w), bias)
predict = ms.ops.Step(output)

# 一个mini batch的样本的损失函数
loss = ms.ops.loss.PerceptionLoss(ms.ops.Multiply(label, output))

# 一个mini batch的平均损失
B =  ms.core.Variable(dim=(1, batch_size), init=False, trainable=False)
B.set_value(1 / batch_size * np.mat(np.ones(batch_size)))
mean_loss = ms.ops.MatMul(B, loss)

# 学习率
learning_rate = 0.0001

# 训练
for epoch in range(50):

    # 遍历训练集中的样本
    for i in np.arange(0, len(train_set), batch_size):

        # 取一个mini batch的样本的特征
        features = np.mat(train_set[i:i + batch_size, :-1])

        # 取一个mini batch的样本的标签
        l = np.mat(train_set[i:i + batch_size, -1]).T

        # 将特征赋给X节点，将标签赋给label节点
        X.set_value(features)
        label.set_value(l)

        # 在平均损失节点上执行前向传播
        mean_loss.forward()

        # 在参数节点上执行反向传播
        w.backward(mean_loss)
        b.backward(mean_loss)

        # 更新参数
        w.set_value(w.value - learning_rate * w.jacobi.T.reshape(w.shape()))
        b.set_value(b.value - learning_rate * b.jacobi.T.reshape(b.shape()))

        ms.default_graph.clear_jacobi()

    # 每个epoch结束后评价模型的正确率
    pred = []

    # 遍历训练集，计算当前模型对每个样本的预测值
    for i in np.arange(0, len(train_set), batch_size):

        features = np.mat(train_set[i:i + batch_size, :-1])
        X.set_value(features)

        # 在模型的predict节点上执行前向传播
        predict.forward()
        
        # 当前模型对一个mini batch的样本的预测结果
        pred.extend(predict.value.A.ravel())

    pred = np.array(pred) * 2 - 1
    accuracy = (train_set[:, -1] == pred).astype(np.int).sum() / len(train_set)
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))
