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

# 类别标签，1男，-1女
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

# 学习率
learning_rate = 0.0001

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
        
        # 在loss节点上执行前向传播，计算损失值
        loss.forward()
        
        # 在w和b节点上执行反向传播，计算损失值对它们的雅克比矩阵
        w.backward(loss)
        b.backward(loss)
        
        """
        用损失值对w和b的雅克比矩阵（梯度的转置）更新参数值。我们欲优化的节点
        都应该是标量节点（才有所谓降低其值一说），它对变量节点的雅克比矩阵都
        是1 x n的，该雅克比的转置是结果节点对变量节点的梯度。将梯度再rehape
        成变量的形状，这样对应位置上就是结果节点对变量元素的偏导数，就可以用来
        更新变量值了。将改变形状后的梯度乘上学习率，从当前变量值中减去，再赋值
        给该变量节点，完成梯度下降更新。
        """
        w.set_value(w.value - learning_rate * w.jacobi.T.reshape(w.shape()))
        b.set_value(b.value - learning_rate * b.jacobi.T.reshape(b.shape()))
        
        # 计算图对象保存了所有节点，在计算图上调用clear_jacobi方法清除所有节点的雅克比矩阵
        ms.default_graph.clear_jacobi()

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

    