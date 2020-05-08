# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 14:47:25 CST 2019

@author: chenzhen
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import grpc
import matrixslow as ms

from .proto import serving_pb2, serving_pb2_grpc


class MatrixSlowServingService(serving_pb2_grpc.MatrixSlowServingServicer):
    '''
    推理服务，主要流程：
    1. 根据模型文件中定义的接口签名，从计算图中获取输入和输出节点
    2. 接受网络请求并解析出模型输入
    3. 调用计算图进行计算
    4. 获取输出节点的值并返回给接口调用者
    '''

    def __init__(self, root_dir, model_file_name, weights_file_name):

        self.root_dir = root_dir
        self.model_file_name = model_file_name
        self.weights_file_name = weights_file_name

        saver = ms.trainer.Saver(self.root_dir)

        # 从文件中加载并还原计算图和参数，同时获取服务接口签名
        _, service = saver.load(model_file_name=self.model_file_name,
                                weights_file_name=self.weights_file_name)
        assert service is not None

        inputs = service.get('inputs', None)
        assert inputs is not None

        outputs = service.get('outputs', None)
        assert outputs is not None

        # 根据服务签名中记录的名字，从计算图中查找出输入和输出节点
        self.input_node = ms.get_node_from_graph(inputs['name'])
        assert self.input_node is not None
        assert isinstance(self.input_node, ms.Variable)

        self.input_dim = self.input_node.dim

        self.output_node = ms.get_node_from_graph(outputs['name'])
        assert self.output_node is not None


    def Predict(self, predict_req, context):

        # 从protobuf数据反序列化成Numpy Matrix
        inference_req = MatrixSlowServingService.deserialize(predict_req)

        # 调用计算图，前向传播计算模型预测结果
        inference_resp = self._inference(inference_req)

        # 将预测结果序列化成protobuf格式，通过网络返回
        predict_resp = MatrixSlowServingService.serialize(inference_resp)

        return predict_resp

    @staticmethod
    def deserialize(predict_req):

        infer_req_mat_list = []
        for proto_mat in predict_req.data:
            dim = tuple(proto_mat.dim)
            mat = np.mat(proto_mat.value, dtype=np.float32)
            mat = np.reshape(mat, dim)
            infer_req_mat_list.append(mat)

        return infer_req_mat_list

    @staticmethod
    def serialize(inference_resp):

        resp = serving_pb2.PredictResp()
        for mat in inference_resp:
            proto_mat = resp.data.add()
            proto_mat.value.extend(np.array(mat).flatten())
            proto_mat.dim.extend(list(mat.shape))

        return resp


    def _inference(self, inference_req):

        inference_resp_mat_list = []

        for mat in inference_req:
            # 将数据输入模型并执行前向传播
            self.input_node.set_value(mat.T)
            self.output_node.forward()

            # 把输出节点的值作为结果返回
            inference_resp_mat_list.append(self.output_node.value)

        return inference_resp_mat_list


class MatrixSlowServer(object):

    def __init__(self, host, root_dir, model_file_name, weights_file_name, max_workers=10):

        self.host = host
        self.max_workers = max_workers

        self.server = grpc.server(
            ThreadPoolExecutor(max_workers=self.max_workers))

        serving_pb2_grpc.add_MatrixSlowServingServicer_to_server(
            MatrixSlowServingService(root_dir, model_file_name, weights_file_name), self.server)

        self.server.add_insecure_port(self.host)


    def serve(self):

        # 启动 rpc 服务
        self.server.start()
        print('MatrixSlow server running on {}'.format(self.host))

        try:
            while True:
                time.sleep(60 * 60 * 24)  # one day in seconds
        except KeyboardInterrupt:
            self.server.stop(0)
