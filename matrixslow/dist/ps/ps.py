# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 17:14:36 CST 2019

@author: chenzhen
"""
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import threading
from core import Node

import grpc
from dist.proto import parameter_server_pb2 as pspb
from dist.proto import parameter_server_pb2_grpc as psrpc


class ParameterServiceServer(psrpc.ParameterServiceServicer):

    def __init__(self, worker_num):
        self.node_gradients_cache = dict()
        self.worker_num = worker_num
        self.cur_push_num = 0
        self.cur_pull_num = self.worker_num
        self.cond = threading.Condition()
        self.acc_no = 0

    @staticmethod
    def _serialize_proto_node_gradients(node_gradients_dict):
        proto_node_gradients = pspb.NodeGradients()
        proto_gradient_total_cost = 0
        proto_gradient_total_cost2 = 0
        proto_gradient_dim_total_cost = 0
        for name, g in node_gradients_dict.items():
            proto_node = proto_node_gradients.nodes.add()
            if isinstance(name, Node):
                name = name.name
            proto_node.name = name
            proto_gradient = proto_node_gradients.gradients.add()

            proto_gradient.value.extend(np.array(g).flatten())
            proto_gradient.dim.extend(list(g.shape))

        return proto_node_gradients

    @staticmethod
    def _deserialize_proto_node_gradients(node_gradients):
        proto_nodes = node_gradients.nodes
        proto_gradients = node_gradients.gradients

        assert len(proto_nodes) == len(proto_gradients)

        node_with_gradients = dict()

        for index in range(len(proto_nodes)):
            node_name = proto_nodes[index].name
            gradients_value = proto_gradients[index].value
            gradients_dim = tuple(proto_gradients[index].dim)
            gradient_mat = np.mat(gradients_value, dtype=np.float32)
            gradient_mat = np.reshape(gradient_mat, gradients_dim)
            node_with_gradients[node_name] = gradient_mat
        return node_with_gradients

    def _deserialize_push_req(self, push_req):
        token = push_req.token
        acc_no = push_req.node_gradients.acc_no
        node_with_gradients = ParameterServiceServer._deserialize_proto_node_gradients(
            push_req.node_gradients)

        return node_with_gradients, acc_no

    def _serialize_pull_resp(self):

        proto_node_gradients = ParameterServiceServer._serialize_proto_node_gradients(
            self.node_gradients_cache)
        resp = pspb.ParameterPullResp(
            token=1, node_gradients=proto_node_gradients)
        return resp

    def _update_gradients_cache(self, node_with_gradients):
        for node_name, gradient in node_with_gradients.items():
            if node_name in self.node_gradients_cache:
                exists_gradient = self.node_gradients_cache[node_name]
                assert exists_gradient.shape == gradient.shape
                self.node_gradients_cache[node_name] = exists_gradient + gradient
            else:
                self.node_gradients_cache[node_name] = gradient

    def _gradients_cache_mean(self):

        if self.acc_no != 0:
            for name, gradient in self.node_gradients_cache.items():
                self.node_gradients_cache[name] = self.node_gradients_cache[name] / self.acc_no
        self.acc_no = 0

    def Push(self, push_req, context):
        # print('Get push req', self.cur_push_num, self.cur_pull_num, context)
        node_with_gradients, acc_no = self._deserialize_push_req(push_req)
        start = time.time()
        if self.cond.acquire():

            while self.cur_pull_num != self.worker_num:
                self.cond.wait()

            self.cur_push_num += 1

            self._update_gradients_cache(node_with_gradients)
            self.acc_no += acc_no
            if self.cur_push_num >= self.worker_num:
                self.cur_pull_num = 0
                self.cond.notify_all()

            self.cond.release()
        else:
            self.cond.wait()
        return pspb.ParameterPushResp(token=123)

    def Pull(self, pull_req, context):
        if self.cond.acquire():
            while self.cur_push_num != self.worker_num:
                self.cond.wait()

            self.cur_pull_num += 1
            self._gradients_cache_mean()
            resp = self._serialize_pull_resp()
            if self.cur_pull_num >= self.worker_num:
                self.cur_push_num = 0
                self.cond.notify_all()

            self.cond.release()
        else:
            self.cond.wait()
        return resp


class ParameterServiceClient():
    def __init__(self, ps_host):
        # 创建stub
        self.stub = psrpc.ParameterServiceStub(
            grpc.insecure_channel(ps_host))

    def push_gradients(self, acc_gradients, acc_no):
        proto_node_gradients = ParameterServiceServer._serialize_proto_node_gradients(
            acc_gradients)
        proto_node_gradients.acc_no = acc_no
        push_req = pspb.ParameterPushReq(
            token=1, node_gradients=proto_node_gradients)

        resp = self.stub.Push(push_req)
        return resp

    def pull_gradients(self, nodes_name):
        pull_req = pspb.ParameterPullReq()

        for node_name in nodes_name:
            proto_node = pull_req.nodes.add()
            proto_node.name = node_name

        pull_resp = self.stub.Pull(pull_req)
        node_gradients_dict = ParameterServiceServer._deserialize_proto_node_gradients(
            pull_resp.node_gradients)
        return node_gradients_dict


def serve(host, worker_num):
    # 启动 rpc 服务
    server = grpc.server(ThreadPoolExecutor(max_workers=10))
    psrpc.add_ParameterServiceServicer_to_server(
        ParameterServiceServer(worker_num), server)
    print('Parameter server running on {} and worker num {}'.format(host, worker_num))
    server.add_insecure_port(host)
    server.start()
    try:
        while True:
            time.sleep(60*60*24)  # one day in seconds
    except KeyboardInterrupt:
        server.stop(0)
