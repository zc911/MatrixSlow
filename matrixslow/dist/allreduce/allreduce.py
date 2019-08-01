# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 12:18:05 CST 2019

@author: chenzhen
"""
import threading
from concurrent.futures import ThreadPoolExecutor

import grpc
from dist.dist import DistCommon
from dist.proto import allreduce_pb2 as arpb
from dist.proto import allreduce_pb2_grpc as arrpc


class RingAllReduceServer(object):
    def __init__(self, host, worker_index, scatter_fn, gather_fn, max_threads=10):
        self.worker_index = worker_index
        self.host = host

        self.server = grpc.server(ThreadPoolExecutor(max_workers=max_threads))
        arrpc.add_RingAllReduceServiceServicer_to_server(
            RingAllReduceService(scatter_fn, gather_fn), self.server)

        self.server.add_insecure_port(self.host)

    def serve(self, daemon=True):
        self.server.start()
        print(
            'Ring All-Reduce worker {} listening on {}'.format(self.worker_index, self.host))
        if not daemon:
            try:
                while True:
                    time.sleep(60*60*24)  # one day in seconds
            except KeyboardInterrupt:
                self.server.stop(0)


class RingAllReduceService(arrpc.RingAllReduceServiceServicer):
    def __init__(self, scatter_fn, gather_fn):
        self.scatter_fn = scatter_fn
        self.gather_fn = gather_fn

    def VariableWeightsInit(self, varibale_weights_req, context):
        pass

    def Recieve(self, send_req, context):
        stage = send_req.stage
        node_gradients_dict = DistCommon._deserialize_proto_node_gradients(
            send_req.node_gradients)
        acc_no = send_req.node_gradients.acc_no

        if stage == arpb.RingAllReduceReq.SCATTER:
            self.scatter_fn(node_gradients_dict, acc_no)
        elif stage == arpb.RingAllReduceReq.GATHER:
            self.gather_fn(node_gradients_dict)
        else:
            print(
                'Invalid ring all-reduce stage: {}, it should be either SCATTER or GATHER'.format(stage))


class RingAllReduceClient(object):
    def __init__(self, target_host):
        self.channel = grpc.insecure_channel(target_host)
        try:
            grpc.channel_ready_future(self.channel).result(timeout=10)
        except grpc.FutureTimeoutError:
            print("!!!!!!!!!!!!!")
        self.stub = arrpc.RingAllReduceServiceStub(self.channel)
        print('Connect to target worker {}'.format(target_host))
        assert self.stub is not None

    def send(self, node_gradients_dict, acc_no, stage):

        proto_node_gradients = DistCommon._serialize_proto_node_gradients(
            node_gradients_dict)
        proto_node_gradients.acc_no = acc_no
        if stage == 'scatter':
            stage = arpb.RingAllReduceReq.SCATTER
        elif stage == 'gather':
            stage = arpb.RingAllReduceReq.GATHER
        else:
            print(
                'Invalid ring all-reduce stage: {}, it should be either SCATTER or GATHER'.format(stage))
        send_req = arpb.RingAllReduceReq(
            stage=stage, node_gradients=proto_node_gradients)
        print('try to send....')
        resp = self.stub.Recieve(send_req)
        print('finish to send....')

        print(resp)
