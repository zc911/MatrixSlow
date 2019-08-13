# -*- coding: utf-8 -*-

"""
Created on Tue Aug 13 11:34:15 CST 2019

@author: chenzhen
"""
import random
import sys
sys.path.append('.')
sys.path.append('../')

import matplotlib
matplotlib.use('TkAgg')

import grpc
import matrixslow_serving as mss
from matrixslow.util import *
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


def plot_img_and_result(img, pred, gt):
    import matplotlib.pyplot as plt
    img = np.reshape(img, (28, 28))
    plt.imshow(img, cmap='Greys_r')

    plt.title('Prediction: {} and Label: {}'.format(pred, gt), color='green' if pred == gt else 'red')
    plt.show(block=False)
    plt.pause(2)


if __name__ == '__main__':
    host = 'localhost:5000'
    client = MatrixSlowServingClient(host)

    _, _, test_x, test_y = util.mnist('../dataset/MNIST')

    while True:
        rand_index = random.randrange(0, len(test_x))
        img = test_x[rand_index]
        label = test_y[rand_index]
        print('Send prediction request...')
        resp = client.Predict([img])
        print('Get Prediction results...')
        resp_mat_list = []
        for proto_mat in resp.data:
            dim = tuple(proto_mat.dim)
            mat = np.mat(proto_mat.value, dtype=np.float32)
            mat = np.reshape(mat, dim)
            resp_mat_list.append(mat)
        pred = np.argmax(resp_mat_list[0])
        gt = np.argmax(label)
        plot_img_and_result(img, pred, gt)
