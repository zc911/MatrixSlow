# -*- coding: utf-8 -*-
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import grpc
import matrixslow as ms

from .proto import serving_pb2, serving_pb2_grpc


class MatrixSlowServingService(serving_pb2_grpc.MatrixSlowServingServicer):
    def __init__(self, root_dir, model_file_name, weights_file_name):
        self.root_dir = root_dir
        self.model_file_name = model_file_name
        self.weights_file_name = weights_file_name

        saver = ms.trainer.Saver(self.root_dir)
        _, service = saver.load(model_file_name=self.model_file_name,
                                weights_file_name=self.weights_file_name)
        assert service is not None

        inputs = service.get('inputs', None)
        assert inputs is not None
        outputs = service.get('outputs', None)
        assert outputs is not None

        self.input_node = ms.get_node_from_graph(inputs['name'])
        assert self.input_node is not None
        assert isinstance(self.input_node, ms.Variable)
        self.input_dim = self.input_node.dim

        self.output_node = ms.get_node_from_graph(outputs['name'])
        assert self.output_node is not None

    def Predict(self, predict_req, context):
        inference_req = MatrixSlowServingService.deserialize(predict_req)
        inference_resp = self._inference(inference_req)
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
            self.input_node.set_value(mat.T)
            self.output_node.forward()
            inference_resp_mat_list.append(self.output_node.value)
        return inference_resp_mat_list


class MatrixSlowServing(object):
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
                time.sleep(60*60*24)  # one day in seconds
        except KeyboardInterrupt:
            self.server.stop(0)
