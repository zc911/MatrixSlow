from ..core import name_scope
from ..core.node import *
from ..ops import Add, MatMul


def logistic_regression(input_size, classes):
    """
    构造多分类逻辑回归模型的计算图.
    :param input_size: 输入维数，即特征数
    :param classes: 类别数
    :return: x: 输入变量，logits: 多分类 logits
    """

    with name_scope('Input'):
        x = Variable((input_size, 1), init=False, trainable=False)

    with name_scope('Parameter'):
        w = Variable((classes, input_size), init=True, trainable=True)
        b = Variable((classes, 1), init=True, trainable=True)

    with name_scope('Logits'):
        logits = Add(MatMul(w, x), b)

    return x, logits