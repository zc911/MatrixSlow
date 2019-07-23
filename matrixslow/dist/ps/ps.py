# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 17:14:36 CST 2019

@author: chenzhen
"""
import time
from concurrent.futures import ThreadPoolExecutor

import grpc
from dist.proto import parameter_server_pb2 as pspb
from dist.proto import parameter_server_pb2_grpc as psrpc


class ParameterServiceServer(psrpc.ParameterServiceServicer):

    def Push(self, push_req, context):
        print('Get push req', context)
        token = push_req.token
        node_gradients = push_req.node_gradients
        nodes = node_gradients.nodes
        gradients = node_gradients.gradients
        print('Workser push...')
        print(token, nodes, gradients)
        return pspb.ParameterPushResp(token=123)

    def Pull(self, pull_req, context):
        print('Get pull req', context)
        token = pull_req.token
        nodes = pull_req.nodes
        gradients = []
        print('Worker pull...')
        print(token, nodes, gradients)
        node_gradients = pspb.NodeGradients(nodes, gradients)
        return pspb.ParameterPullResp(token, node_gradients)


class ParameterServiceClient():
    def __init__(self, ip, port):
        # 创建stub
        self.stub = psrpc.ParameterServiceStub(
            grpc.insecure_channel('{}:{}'.format(ip, port)))

    def push_gradients(self, acc_gradient):
        node_gradients = pspb.NodeGradients()
        for n, g in acc_gradient.items():
            node = node_gradients.nodes.add()
            node.name = n.name
            node.node_type = n.__class__.__name__

            gradient = node_gradients.gradients.add()
            gradient.value = g.flatten().tolist()
            gradient.dim = list(g.shape)
        req = pspb.ParameterPushReq(token=1, node_gradients=node_gradients)
        resp = self.stub.Push(req)
        print(resp)

    def pull_gradients(self, nodes_name):
        pass


def serve():
    # 启动 rpc 服务
    server = grpc.server(ThreadPoolExecutor(max_workers=10))
    psrpc.add_ParameterServiceServicer_to_server(
        ParameterServiceServer(), server)
    print('Parameter server running on [::]:50051')
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(60*60*24)  # one day in seconds
    except KeyboardInterrupt:
        server.stop(0)
