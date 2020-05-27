#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: chenzhen
@Date: 2020-04-22 15:06:58
@LastEditTime: 2020-04-22 15:49:31
@LastEditors: chenzhen
@Description:
'''

import sys
sys.path.append('../../')

import grpc
import numpy as np
from sklearn.datasets import fetch_openml

import matrixslow_serving as mss
from matrixslow_serving.serving import serving_pb2, serving_pb2_grpc



class MatrixSlowServingClient(object):
    def __init__(self, host):
        self.stub = serving_pb2_grpc.MatrixSlowServingStub(
            grpc.insecure_channel(host))
        print('[GRPC] Connected to MatrixSlow serving: {}'.format(host))

    def Predict(self, mat_data_list):
        req = serving_pb2.PredictReq()
        for mat in mat_data_list:
            proto_mat = req.data.add()
            proto_mat.value.extend(np.array(mat).flatten())
            proto_mat.dim.extend(list(mat.shape))
        resp = self.stub.Predict(req)
        return resp


if __name__ == '__main__':
    # 加载MNIST数据集，取一部分样本并归一化
    img_shape = (28, 28)
    test_data, test_label = fetch_openml(
        'mnist_784', version=1, return_X_y=True, cache=True)
    test_data, test_label = test_data[1000:2000] / \
        255, test_label.astype(np.int)[1000:2000]
    test_data = np.reshape(np.array(test_data), (1000, *img_shape))

    host = '127.0.0.1:5000'
    client = MatrixSlowServingClient(host)

    for index in range(len(test_data)):
        img = test_data[index]
        label = test_label[index]
        resp = client.Predict([img])
        resp_mat_list = []
        for proto_mat in resp.data:
            dim = tuple(proto_mat.dim)
            mat = np.mat(proto_mat.value, dtype=np.float32)
            mat = np.reshape(mat, dim)
            resp_mat_list.append(mat)
        pred = np.argmax(resp_mat_list[0])
        gt = label
        print('model predict {} and ground truth: {}'.format(
            np.argmax(pred.value), gt))
