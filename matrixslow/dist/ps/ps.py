# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 17:14:36 CST 2019

@author: chenzhen
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import grpc
from core import Node
from dist import DistCommon
from dist.proto import parameter_server_pb2 as pspb
from dist.proto import parameter_server_pb2_grpc as psrpc


class ParameterService(psrpc.ParameterServiceServicer):
    '''
    简单的Parameter Server，支持同步和异步两种更新模式。
    A 严格的同步更新梯度模式，即：
    1. 所有的worker都把梯度push到ps
    2. 所有的worker都把梯度pull到本地

    B 异步更新梯度模式，即：
    所有的worker按照随机的顺序访问PS，把自身的梯度更新到PS或从PS拉取最新的平均梯度
    worker间仅通过一个锁保证数据读写的正确性，而不保证完整性
    '''

    def __init__(self, worker_num, sync=True):
        self.node_gradients_cache = dict()
        self.variable_weights_cache = dict()
        # PS运行同步还是异步模式
        self.sync = sync
        self.worker_num = worker_num
        self.cur_push_num = 0
        self.cur_pull_num = self.worker_num

        self.cond = threading.Condition()
        self.push_lock = threading.Lock()
        self.init_lock = threading.Lock()
        self.is_init = False

        self.acc_no = 0

    def _deserialize_push_req(self, push_req):
        '''
        反序列化push request
        '''
        acc_no = push_req.node_gradients.acc_no
        node_with_gradients = DistCommon._deserialize_proto_node_gradients(
            push_req.node_gradients)

        return node_with_gradients, acc_no

    def _serialize_pull_resp(self):
        '''
        序列化pull response
        '''
        proto_node_gradients = DistCommon._serialize_proto_node_gradients(
            self.node_gradients_cache)
        resp = pspb.ParameterPullResp(node_gradients=proto_node_gradients)
        return resp

    def _update_gradients_cache(self, node_with_gradients):
        '''
        按照变量名/节点名，更新缓存的梯度值
        '''
        for node_name, gradient in node_with_gradients.items():
            if node_name in self.node_gradients_cache:
                exists_gradient = self.node_gradients_cache[node_name]
                assert exists_gradient.shape == gradient.shape
                self.node_gradients_cache[node_name] = exists_gradient + gradient
            else:
                self.node_gradients_cache[node_name] = gradient

    def _gradients_cache_mean(self):
        '''
        对缓存的梯度值进行平均操作
        '''
        if self.acc_no != 0:
            for name, gradient in self.node_gradients_cache.items():
                self.node_gradients_cache[name] = self.node_gradients_cache[name] / self.acc_no
            self.acc_no = 0

    def _reset_gradients_cache(self):
        self.node_gradients_cache.clear()

    def _push_sync(self, node_with_gradients, acc_no):
        '''
        同步模式的push操作
        '''
        if self.cond.acquire():

            while self.cur_pull_num != self.worker_num:
                self.cond.wait()

            self.cur_push_num += 1

            self._update_gradients_cache(node_with_gradients)
            # 累计梯度数量
            self.acc_no += acc_no
            # 如果所有worker都push梯度完成，通知所有worker从ps pull梯度
            if self.cur_push_num >= self.worker_num:
                self.cur_pull_num = 0
                self.cond.notify_all()

            self.cond.release()
        else:
            self.cond.wait()

    def _push_async(self,  node_with_gradients, acc_no):
        '''
        异步模式的push操作
        '''
        self.push_lock.acquire()
        self._update_gradients_cache(node_with_gradients)
        # 累计梯度数量
        self.acc_no += acc_no
        self.push_lock.release()

    def Push(self, push_req, context):
        '''
        Push梯度到Parameter server并更新
        '''
        node_with_gradients, acc_no = self._deserialize_push_req(push_req)
        if self.sync:
            self._push_sync(node_with_gradients, acc_no)
        else:
            self._push_async(node_with_gradients, acc_no)

        return pspb.ParameterPushResp()

    def _pull_sync(self):
        '''
        同步模式的pull操作
        '''
        if self.cond.acquire():
            while self.cur_push_num != self.worker_num:
                self.cond.wait()

            self.cur_pull_num += 1
            self._gradients_cache_mean()
            resp = self._serialize_pull_resp()
            # 如果所有worker都已完成pull，通知worker开始push梯度
            if self.cur_pull_num >= self.worker_num:
                self.cur_push_num = 0
                self._reset_gradients_cache()
                self.cond.notify_all()

            self.cond.release()
        else:
            self.cond.wait()
        return resp

    def _pull_async(self):
        '''
        异步模式的pull操作
        '''
        self.push_lock.acquire()
        self._gradients_cache_mean()
        resp = self._serialize_pull_resp()
        self._reset_gradients_cache()
        self.push_lock.release()

        return resp

    def Pull(self, pull_req, context):
        '''
        从PS中pull梯度
        '''
        if self.sync:
            resp = self._pull_sync()
        else:
            resp = self._pull_async()
        return resp

    def VariableWeightsInit(self, varibale_weights_req, context):
        '''
        权值变量初始化接口。多个worker同时把自身变量发送到PS，
        PS简单的使用第一个到达的worker的权值变量作为初始值，更新到其他worker中
        '''
        self.init_lock.acquire()
        # 如果未初始化，使用第一个worker的权值变量
        if not self.is_init:
            self.variable_weights_cache = DistCommon._deserialize_proto_variable_weights(
                varibale_weights_req)
        # 其他worker使用已经存在的权值变量，更新自身
        resp = DistCommon._serialize_proto_variable_weights(
            self.variable_weights_cache)
        self.is_init = True
        self.init_lock.release()
        return resp


class ParameterServiceClient(object):
    '''
    Parameter Server的client帮助类
    '''

    def __init__(self, ps_host):
        # 创建grpc stub
        self.stub = psrpc.ParameterServiceStub(
            grpc.insecure_channel(ps_host))
        assert self.stub is not None

    def variable_weights_init(self, var_weights_dict):
        init_req = DistCommon._serialize_proto_variable_weights(
            var_weights_dict)
        init_resp = self.stub.VariableWeightsInit(init_req)
        duplicated_var_weights_dict = DistCommon._deserialize_proto_variable_weights(
            init_resp)
        return duplicated_var_weights_dict

    def push_gradients(self, acc_gradients, acc_no):
        '''
        执行梯度push操作
        '''
        # 把梯度序列化为protobuf对象
        proto_node_gradients = DistCommon._serialize_proto_node_gradients(
            acc_gradients)
        # 当前梯度的累计数量
        proto_node_gradients.acc_no = acc_no
        # 构造并发送push网络请求
        push_req = pspb.ParameterPushReq(node_gradients=proto_node_gradients)
        resp = self.stub.Push(push_req)
        return resp

    def pull_gradients(self, nodes_name=None):
        '''
        执行梯度pull操作
        '''
        # 构造并发送pull请求，默认情况下，pull所有变量的梯度
        pull_req = pspb.ParameterPullReq()
        # for node_name in nodes_name:
        #     proto_node = pull_req.nodes.add()
        #     proto_node.name = node_name

        pull_resp = self.stub.Pull(pull_req)
        # 把返回protobuf对象结果反序列化
        node_gradients_dict = DistCommon._deserialize_proto_node_gradients(
            pull_resp.node_gradients)
        return node_gradients_dict


class ParameterServiceServer(object):
    def __init__(self, cluster_conf, sync=True, max_workers=10):

        self.worker_num = len(cluster_conf['workers'])
        self.host = cluster_conf['ps'][0]
        self.sync = sync
        self.max_workers = max_workers

        self.server = grpc.server(
            ThreadPoolExecutor(max_workers=self.max_workers))
        psrpc.add_ParameterServiceServicer_to_server(
            ParameterService(self.worker_num, self.sync), self.server)
        self.server.add_insecure_port(self.host)

    def serve(self):
        # 启动 rpc 服务
        self.server.start()
        print('[PS] Parameter server (mode: {}) running on {} and worker num {}'.format('Sync' if self.sync else 'Async',
                                                                                   self.host, self.worker_num))
        try:
            while True:
                time.sleep(60*60*24)  # one day in seconds
        except KeyboardInterrupt:
            self.server.stop(0)
