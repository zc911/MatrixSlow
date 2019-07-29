from ..core import *
from ..ops import *


def conv(feature_maps, input_shape, kernels, kernel_shape, activation):
    """
    :param feature_maps:
    :param input_shape:
    :param kernels:
    :param kernel_shape:
    :param activation:
    :return:
    """
    outputs = []
    for i in range(kernels):

        channels = []
        for fm in feature_maps:
            kernel = Variable(kernel_shape, init=True, trainable=True)
            conv = Convolve(fm, kernel)
            channels.append(conv)

        channles = Add(*channels)
        bias = Variable(input_shape, init=True, trainable=True)
        affine = Add(channles, bias)

        if activation == "ReLU":
            outputs.append(ReLU(affine))
        elif activation == "Logistic":
            outputs.append(Logistic(affine))
        else:
            outputs.append(affine)

    assert len(outputs) == kernels
    return outputs


def pooling(feature_maps, kernel_shape, stride):
    """
    :param feature_maps:
    :param kernel_shape:
    :param stride:
    :return:
    """
    outputs = []
    for fm in feature_maps:
        outputs.append(MaxPooling(fm, size=kernel_shape, stride=stride))

    return outputs


def fc(input, input_size, size, activation):
    """
    :param input:
    :param input_size:
    :param size:
    :param activation:
    :return:
    """
    weights = Variable((size, input_size), init=True, trainable=True)
    bias = Variable((size, 1), init=True, trainable=True)
    affine = Add(MatMul(weights, input), bias)

    if activation == "ReLU":
        return ReLU(affine)
    elif activation == "Logistic":
        return Logistic(affine)
    else:
        return affine
