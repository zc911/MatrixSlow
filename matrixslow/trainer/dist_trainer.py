# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 20:48:16 CST 2019

@author: chenzhen
"""

import threading

from core import (Variable, get_trainable_variables_from_graph,
                  update_node_value_in_graph)
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
        self.worker_index = kargs['worker_index']

        self.cur_acc_no = 0
        self.cur_partion_index = self.worker_index
        self.workers = self.cluster_conf['workers']
        self.host = self.workers[self.worker_index]
        self.worker_num = len(self.workers)

        self.step = self.worker_num - 1
        self.target_index = (self.worker_index + 1) % self.worker_num
        self.target_host = self.workers[self.target_index]

        self.server = allreduce.RingAllReduceServer(
            self.host, self.worker_index, self._scatter, self._gather)
        self.client = allreduce.RingAllReduceClient(self.target_host)

        self.partition = []
        self.variables = get_trainable_variables_from_graph()
        self._partition_variables()

        self.is_recieved = False
        self.recieved_gradients = None
        self.recieved_acc_no = None

        self.cond = threading.Condition()

        self.server.serve()

    def _partition_variables(self):
        var_num = len(self.variables)
        part_length = int(var_num / self.worker_num)
        assert part_length > 0
        start = 0
        end = start + part_length
        for i in range(self.worker_num - 1):
            self.partition.append((start, end))
            start = end
            end = start + part_length
        self.partition.append((start, var_num))
        print('self.partition, ', self.partition)

    def _get_gradients_partition(self):
        start, end = self.partition[self.cur_partion_index]
        part_variables = self.variables[start:end]
        self.cur_partion_index = (
            self.cur_partion_index + self.step) % self.worker_num
        part_gradients = dict()
        for var in part_variables:
            part_gradients[var] = self.optimizer.acc_gradient[var]
        return part_gradients

    def _scatter(self, node_gradients_dict, acc_no):
        self.recieved_gradients = node_gradients_dict
        self.recieved_acc_no = acc_no
        self.is_recieved = True
        self.cond.notify_all()

    def _gather(self, node_gradients_dict):
        self.optimizer.apply_gradients(
            node_gradients_dict, gather=True)

    def _wait_for_revieve(self):
        if self.cond.acquire():
            while not self.is_recieved:
                self.cond.wait()
            self.optimizer.apply_gradients(
                self.recieved_gradients, acc_no=self.recieved_acc_no, summarize=True)
            self.is_recieved = False
            self.cond.release()
        else:
            self.cond.wait()

    def _optimizer_update(self):
        # the scatter operation
        for scatter_index in range(self.step):
            gradients_part = self._get_gradients_partition()
            cur_acc_no = self.optimizer.acc_no if self.recieved_acc_no is None else self.recieved_acc_no
            self.client.send(gradients_part, cur_acc_no, 'scatter')
            self._wait_for_recieve()

        # the all gather operation
        # for gather_index in range(self.step):
        #     self.client.send()
        #     self._wait_for_revieve()

        self.optimizer.update()
