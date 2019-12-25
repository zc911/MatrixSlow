import numpy as np
import matrixslow as ms

# 输入向量，是一个3x1矩阵，不需要初始化，不参与训练
x = ms.core.Variable(dim=(3, 1), init=False, trainable=False)

# 类别标签，男 1 女 -1
label = ms.core.Variable(dim=(1, 1), init=False, trainable=False)

# 权重向量，是一个1x3矩阵，需要初始化，参与训练
w = ms.core.Variable(dim=(1, 3), init=True, trainable=True)

# 阈值，是一个1x1矩阵，需要初始化，参与训练
b = ms.core.Variable(dim=(1, 1), init=True, trainable=True)

# 构造ADALINE的计算图
output = ms.ops.Add(ms.ops.MatMul(w, x), b)
predict = ms.ops.Step(output)

# 损失函数
loss = ms.ops.loss.PerceptionLoss(ms.ops.MatMul(label, output))

# 对一个样本计算损失值和输出，首先将样本赋给x变量
x.set_value(np.mat([182, 72, 0.17]).T)
label.set_value(np.mat([[1]]))

# 计算模型的输出
predict.forward()
loss.forward()
print("该样本为{:s}， 损失值是{:.8f}".format(
    "男士" if predict.value[0, 0] == 1.0 else "女士",
    loss.value[0, 0])
)