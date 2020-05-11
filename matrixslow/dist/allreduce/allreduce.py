# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 12:18:05 CST 2019

@author: chenzhen
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import grpc
from ...core import default_graph
from ..dist import DistCommon
from ..proto import allreduce_pb2 as arpb
from ..proto import allreduce_pb2_grpc as arrpc
from ..proto import common_pb2


class RingAllReduceServer(object):

    def __init__(self, host, worker_index, vars_init_fn, scatter_fn, gather_fn, max_threads=10):
        self.worker_index = worker_index
        self.host = host

        self.server = grpc.server(ThreadPoolExecutor(max_workers=max_threads))
        arrpc.add_RingAllReduceServiceServicer_to_server(
            RingAllReduceService(vars_init_fn, scatter_fn, gather_fn), self.server)

        self.server.add_insecure_port(self.host)

    def _serve(self):
        self.server.start()
        print(
            '[GRPC] Ring All-Reduce worker {} listening on {}'.format(self.worker_index, self.host))
        try:
            while True:
                time.sleep(60*60*24)  # one day in seconds
        except KeyboardInterrupt:
            self.server.stop(0)

    def serve(self):
        threading.Thread(target=self._serve).start()


class RingAllReduceService(arrpc.RingAllReduceServiceServicer):

    def __init__(self, vars_init_fn, scatter_fn, gather_fn):

        # 参数初始化回调函数，由外部trainer传入
        self.vars_init_fn = vars_init_fn

        # scatter回调函数，由外部的trainer传入
        self.scatter_fn = scatter_fn

        # gather回调函数，由外部的trainer传入
        self.gather_fn = gather_fn


    def VariableWeightsInit(self, varibale_weights_req, context):
        '''
        变量节点初始化。接收上一个worker发送来的初始值并更新自身的变量节点值
        '''
        variable_weights_cache = DistCommon._deserialize_proto_variable_weights(
            varibale_weights_req)
        self.vars_init_fn(variable_weights_cache)

        return common_pb2.VariableWeightsReqResp()


    def Recieve(self, send_req, context):
        stage = send_req.stage

        # 从gRPC请求中解析出发送来的节点和梯度
        node_gradients_dict = DistCommon._deserialize_proto_node_gradients(
            send_req.node_gradients)

        # 接收到左邻居的请求，根据当前阶段的不同，执行不同的回调函数
        if stage == arpb.RingAllReduceReq.SCATTER:
            acc_no = send_req.node_gradients.acc_no
            self.scatter_fn(node_gradients_dict, acc_no)
        elif stage == arpb.RingAllReduceReq.GATHER:
            self.gather_fn(node_gradients_dict)
        else:
            print(
                '[ALLREDUCE] Invalid ring all-reduce stage: {}, it should be either SCATTER or GATHER'.format(stage))

        return arpb.RingAllReduceResp()


class RingAllReduceClient(object):
    def __init__(self, target_host, timeout=30):
        self.timeout = timeout
        try:
            print('[GRPC] Try connect to target worker {}'.format(target_host))
            self.channel = grpc.insecure_channel(target_host)
            grpc.channel_ready_future(
                self.channel).result(timeout=self.timeout)
        except grpc.FutureTimeoutError:
            print("[GRPC] Failed connect to target worker")
            assert 0
        else:
            self.stub = arrpc.RingAllReduceServiceStub(self.channel)
            print('[GRPC] Connected to target worker {}'.format(target_host))
            assert self.stub is not None

    def variable_weights_init(self, var_weights_dict):
        init_req = DistCommon._serialize_proto_variable_weights(
            var_weights_dict)
        resp = self.stub.VariableWeightsInit(init_req)

    def send(self, node_gradients_dict, acc_no, stage):

        proto_node_gradients = DistCommon._serialize_proto_node_gradients(
            node_gradients_dict)

        if stage == 'scatter':
            proto_node_gradients.acc_no = acc_no
            stage = arpb.RingAllReduceReq.SCATTER
        elif stage == 'gather':
            stage = arpb.RingAllReduceReq.GATHER
        else:
            print(
                '[ALLREDUCE] Invalid ring all-reduce stage: {}, it should be either SCATTER or GATHER'.format(stage))
        send_req = arpb.RingAllReduceReq(
            stage=stage, node_gradients=proto_node_gradients)
        resp = self.stub.Recieve(send_req, timeout=self.timeout)
