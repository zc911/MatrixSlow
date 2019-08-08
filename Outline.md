## 卷一
1. 计算图与反向传播（大约50页）
a)	计算图模型
b)	VectorSlow 框架
2. 反向传播和求导
a)	梯度下降
b)	自动求导的实现原理
3. 损失函数与优化算法（大约40页）
a)	随机梯度下降的原理及实现
b)	Mini Batch梯度下降的原理及实现
c)	AdaGrad的原理及实现
d)	RMSProp的原理及实现
e)	Adam的原理及实现
4. 模型的训练、评估与优化（大约50页）
a)	评价指标
b)	训练集、测试集
c)	自由度与正则化
d)	VectorSlow 的训练器（写一个类似trainer 的东西）
5. 线性回归与逻辑回归（大约30页）
a)	原理
b)	实现
c)	应用
6. 多层全连接神经网络与Word2Vec算法（大约30页）
a)	原理
b)	实现
c)	应用
7. Deep & Wide & FM、FFM与DeepFM（大约30页）
a)	原理
b)	实现
c)	应用
8. 卷积神经网络（大约30页）
a)	原理
b)	实现
c)	应用
## 卷二
9. 模型保存、加载和推理
a)	保存
b)	加载
c)	推理
10. 分布式VectorSlow
a)	基本概念和技术
b)	数据并行与模型并行
c)	PS-Server的实现
d)	Ring All-Reduce的实现
11. DataSlow数据模块
a)	数据模块的意义
b)	采样与增广
c)	设计与实现
12. VectorSlow Serving模型服务框架
a)	概念和技术
b)	设计与实现