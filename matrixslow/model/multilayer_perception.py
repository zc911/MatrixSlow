from ..core import name_scope
from ..core.node import *
from ..layer import fc


def multilayer_perception(input_size, classes, hidden_layers, activation):
    """
    构造多层感知机，即多层全连接神经网络的计算图.
    :param input_size: 输入维数，即特征数
    :param classes: 类别数
    :param hidden_layers: 数组，包含每个隐藏层的神经元数
    :param activation: 指定隐藏层激活函数类型，若为 None 则无激活函数
    :return: x: 输入变量，logits: 多分类 logits
    """

    with name_scope('Input'):
        x = Variable((input_size, 1), init=False, trainable=False)

    with name_scope('Hidden'):

        output = x
        for size in hidden_layers:
            output = fc(output, input_size, size, activation)
            input_size = size

    with name_scope('Logits'):
        logits = fc(output, input_size, classes, None)  # 无激活函数

    return x, logits