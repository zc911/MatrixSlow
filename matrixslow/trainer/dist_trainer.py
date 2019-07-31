# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 20:48:16 CST 2019

@author: chenzhen
"""

from core import Variable, update_node_value_in_graph
from core.graph import default_graph
from dist import allreduce, ps
from trainer import Trainer


class DistTrainerParameterServer(Trainer):
    def __init__(self, *args, **kargs):
        Trainer.__init__(self, *args, **kargs)
        cluster_conf = kargs['cluster_conf']
        ps_host = cluster_conf['ps'][0]
        self.ps_client = ps.ParameterServiceClient(ps_host)

    def _variable_weights_init(self):
        '''
        多个worker通过ps保证权值变量的一致
        '''
        var_weights_dict = dict()
        for node in default_graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                var_weights_dict[node.name] = node.value
        duplicated_var_weights_dict = self.ps_client.variable_weights_init(
            var_weights_dict)
        for var_name, weights in duplicated_var_weights_dict.items():
            update_node_value_in_graph(var_name, weights)

    def _optimizer_update(self):
        # 把当前梯度push到ps上。此操作可能被block，直到所有节点都pull完成
        acc_gradient = self.optimizer.acc_gradient
        self.ps_client.push_gradients(
            acc_gradient, self.optimizer.acc_no)
        # 从ps把所有节点的平均梯度pull回来。此操作可能被block直到所有节点都push完成
        node_gradients_dict = self.ps_client.pull_gradients()
        # 使用平均梯度，更新本地变量
        self.optimizer.update(node_gradients_dict)


class DistTrainerRingAllReduce(Trainer):
    def __init__(self, *args, **kargs):
        Trainer.__init__(self, *args, **kargs)
        self.cluster_conf = kargs['cluster_conf']
        self.index = kargs['index']

        self.acc_no = 0
        self.workers = self.cluster_conf['workers']
        self.host = self.workers[self.index]
        self.worker_num = len(self.workers)
        self.step = self.worker_num - 1
        self.target_index = (self.index + 1) % self.worker_num
        self.target_host = self.workers[self.target_index]

        self.server = allreduce.RingAllReduceServer(
            self.host, self._scatter, self._gather)
        self.client = allreduce.RingAllReduceClient(self.target_host)

        self.server.serve()

    def _scatter(self, node_gradients_dict, acc_no):
        self.acc_no += acc_no
        self.optimizer.apply_gradients(
            node_gradients_dict, gather=False)

    def _gather(self, node_gradients_dict):

        self.optimizer.apply_gradients(
            node_gradients_dict, acc_no=self.acc_no, gather=True)

    def _optimizer_update(self):

        self.optimizer.update()
