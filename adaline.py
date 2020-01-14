import numpy as np
import matrixslow as ms

"""
制造训练样本。根据均值171，标准差6的正态分布采样500个男性身高，根据均值158，
标准差5的正态分布采样500个女性身高。根据均值70，标准差10的正态分布采样500个
男性体重，根据均值57，标准差8的正态分布采样500个女性体重。根据均值16，标准差
2的正态分布采样500个男性体脂率，根据均值22，标准差2的正态分布采样500个女性体
脂率。构造500个1，作为男性标签，构造500个-1，作为女性标签。将数据组装成一个
1000x4的numpy数组，前3列分别是身高、体重和体脂率，最后一列是性别标签。
"""
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
    

# 构造计算图：输入向量，是一个3x1矩阵，不需要初始化，不参与训练
x = ms.core.Variable(dim=(3, 1), init=False, trainable=False)

# 类别标签，男 1 女 -1
label = ms.core.Variable(dim=(1, 1), init=False, trainable=False)

# 权重向量，是一个1x3矩阵，需要初始化，参与训练
w = ms.core.Variable(dim=(1, 3), init=True, trainable=True)

# 阈值，是一个1x1矩阵，需要初始化，参与训练
b = ms.core.Variable(dim=(1, 1), init=True, trainable=True)

# ADALINE的预测输出
output = ms.ops.Add(ms.ops.MatMul(w, x), b)
predict = ms.ops.Step(output)

# 损失函数
loss = ms.ops.loss.PerceptionLoss(ms.ops.MatMul(label, output))

# 优化器
optimizer = ms.optimizer.GradientDescent(ms.default_graph, loss, learning_rate=0.001)

# 训练，批大小为10
batch_size = 10

# 训练执行50个epoch
for epoch in range(50):
    
    # 批计数器清零
    batch_count = 0
    
    # 遍历训练集中的样本
    for i in range(len(train_set)):
        
        # 取第i个样本的前4列（出最后一列的所有列），构造3x1矩阵对象
        features = np.mat(train_set[i,:-1]).T
        
        # 取第i个样本的最后一列，是该样本的性别标签（1男，-1女），构造1x1矩阵对象
        l = np.mat(train_set[i,-1])
        
        # 将特征赋给x变量，将标签赋给label变量
        x.set_value(features)
        label.set_value(l)
        
        # 优化器执行一步，在本样本上完成一次前向传播和反向传播
        optimizer.one_step()
        
        # 批计数器加1
        batch_count += 1
        
        # 若批计数器大于等于批大小，则执行一次梯度下降更新，并清零计数器
        if batch_count >= batch_size:
            optimizer.update()
            batch_count = 0

    # 每个epoch结束后评价一下模型的正确率
    pred = []
    
    # 遍历训练集，计算当前模型对每个样本的预测值
    for i in range(len(train_set)):
                
        features = np.mat(train_set[i,:-1]).T
        x.set_value(features)
        
        # 在模型的predict节点上执行前向传播
        predict.forward()
        pred.append(predict.value[0, 0])  # 模型的预测结果：1男，0女
            
    pred = np.array(pred) * 2 - 1  # 将1/0结果转化成1/-1结果，好与训练标签的约定一致
    
    # 判断预测结果与样本标签相同的数量与训练集总数量之比，即模型预测的正确率
    accuracy = (train_set[:,-1] == pred).astype(np.int).sum() / len(train_set)
    
    
    # 打印当前epoch数和模型在训练集上的正确率
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))

    