# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 20:48:16 CST 2019

@author: chenzhen
"""
import math
import threading
import time

from ..core import (Variable, get_trainable_variables_from_graph,
                    update_node_value_in_graph)
from ..core.graph import default_graph
from ..dist import allreduce, ps
from .trainer import Trainer


class DistTrainerParameterServer(Trainer):

    def __init__(self, *args, **kargs):
        Trainer.__init__(self, *args, **kargs)
        cluster_conf = kargs['cluster_conf']
        ps_host = cluster_conf['ps'][0]
        self.ps_client = ps.ParameterServiceClient(ps_host)


    def _variable_weights_init(self):
        '''
        多个worker通过ps保证变量节点初始化一致
        '''
        var_weights_dict = dict()
        for node in default_graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                var_weights_dict[node.name] = node.value

        # 把自己的初始值发送给ps，由ps决定使用哪个Worker并返回
        duplicated_var_weights_dict = self.ps_client.variable_weights_init(
            var_weights_dict)

        # 使用ps返回的初始值，重新初始化本地
        for var_name, weights in duplicated_var_weights_dict.items():
            update_node_value_in_graph(var_name, weights)

        print('[INIT] Worker variable weights initialized')


    def _optimizer_update(self):

        # 把当前梯度push到ps上。此操作可能被block，直到所有节点都pull完成
        acc_gradient = self.optimizer.acc_gradient
        self.ps_client.push_gradients(
            acc_gradient, self.optimizer.acc_no)

        # 从ps把所有节点的平均梯度pull回来。此操作可能被block直到所有节点都push完成
        node_gradients_dict = self.ps_client.pull_gradients()

        # 使用平均梯度，利用优化器的优化算法，更新本地变量
        self.optimizer.update(node_gradients_dict)


class DistTrainerRingAllReduce(Trainer):
    '''
    Ring All-Reduce模式的分布式训练
    '''

    def __init__(self, *args, **kargs):
        Trainer.__init__(self, *args, **kargs)

        # 读取集群配置信息和自身信息
        self.cluster_conf = kargs['cluster_conf']
        self.worker_index = kargs['worker_index']

        self.workers = self.cluster_conf['workers']
        self.worker_num = len(self.workers)
        self.host = self.workers[self.worker_index]

        self.step = self.worker_num - 1

        # 根据集群的环状拓扑结构确定右邻居
        self.target_host = self.workers[(
            self.worker_index + 1) % self.worker_num]

        # 本节点是否已被初始化
        self.is_init = False
        self.init_cond = threading.Condition()

        self.cur_partion_index = self.worker_index
        self.partition = []

        # 获取所有可训练节点
        self.variables = get_trainable_variables_from_graph()

        # 根据worker的总数量，对即将更新的变量节点列表进行等长切分
        self._partition_variables()

        # 用于控制梯度的发送和接收
        self.is_recieved = False
        self.recieved_gradients = None
        self.recieved_acc_no = None
        self.cond = threading.Condition()

        # 创建本节点的梯度接收服务
        allreduce.RingAllReduceServer(
            self.host, self.worker_index,
            self._variable_weights_init_callback,
            self._scatter_callback,
            self._gather_callback).serve()

        # 创建连接目标节点的梯度发送client
        self.client = allreduce.RingAllReduceClient(self.target_host)


    def _variable_weights_init(self):

        var_weights_dict = dict()
        for node in default_graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                var_weights_dict[node.name] = node.value
        print('[INIT] Send variable init weights to worker ', self.target_host)

        # 第一个节点不需要等待，使用默认值更新给下一个节点
        if self.worker_index == 0:
            self.client.variable_weights_init(var_weights_dict)
        else:
            self.init_cond.acquire()
            while not self.is_init:
                self.init_cond.wait()
            self.init_cond.release()
            self.client.variable_weights_init(var_weights_dict)


    def _variable_weights_init_callback(self, var_weights_dict):

        # 第一个节点不需要接收上一个节点的初始值
        if self.worker_index != 0:
            print('[INIT] Variables initializing weights from last worker node...')
            for var_name, weights in var_weights_dict.items():
                update_node_value_in_graph(var_name, weights)

        # 已初始化完成，通知发送流程
        self.init_cond.acquire()
        self.is_init = True
        self.init_cond.notify_all()
        self.init_cond.release()


    def _optimizer_update(self):

        # 共执行 N-1 次scatter操作，把本worker的梯度切片发送给下一个worker
        # 同时接收左邻居发送过来的梯度，累加到自己的对应切片上
        for scatter_index in range(self.step):
            gradients_part = self._get_gradients_partition()
            cur_acc_no = self.optimizer.acc_no if scatter_index == 0 else self.recieved_acc_no

            # 把自身的一个数据分块发送给右邻居
            self.client.send(gradients_part, cur_acc_no, 'scatter')

            # 等待接收并处理完左邻居节点的数据
            self._wait_for_recieve('scatter')

        # 然后执行 N-1 次all-gather操作，把本worker的梯度切片发送给下一个worker
        # 同时接收上一个worker发送过来的梯度并替换自己的对应切片
        for gather_index in range(self.step):
            gradients_part = self._get_gradients_partition()
            self.client.send(gradients_part, 0, 'gather')
            self._wait_for_recieve('gather')

        self.optimizer.update()


    def _partition_variables(self):
        '''
        根据worker的总数量，对即将更新的权值变量列表进行等长切分
        '''
        var_num = len(self.variables)
        part_length = math.ceil(var_num / self.worker_num)
        assert part_length > 0

        start = 0
        end = start + part_length
        for i in range(self.worker_num - 1):
            self.partition.append((start, end))
            start = end
            end = start + part_length

        self.partition.append((start, var_num))


    def _get_gradients_partition(self):
        '''
        获取下一个梯度切片
        '''
        start, end = self.partition[self.cur_partion_index]
        part_variables = self.variables[start:end]
        self.cur_partion_index = (
            self.cur_partion_index + self.step) % self.worker_num
        part_gradients = dict()
        for var in part_variables:
            part_gradients[var] = self.optimizer.acc_gradient[var]
        return part_gradients


    def _scatter_callback(self, node_gradients_dict, acc_no):
        '''
        Scatter 阶段的回调函数，接收上一个worker发送过来的梯度和样本数
        '''
        if self.cond.acquire():
            while self.is_recieved:
                self.cond.wait()

            # 把接收到的梯度缓存下来
            self.recieved_gradients = node_gradients_dict
            self.recieved_acc_no = acc_no
            self.is_recieved = True

            # 通知主流程，把接收到的梯度更新到优化器
            self.cond.notify_all()
            self.cond.release()
        else:
            self.cond.wait()


    def _gather_callback(self, node_gradients_dict):
        '''
        All-gather 阶段的回调函数，接收上一个worker发送来的梯度
        '''
        if self.cond.acquire():
            while self.is_recieved:
                self.cond.wait()

            self.recieved_gradients = node_gradients_dict
            self.is_recieved = True

            # 通知主流程，把接收到的梯度更新到优化器
            self.cond.notify_all()
            self.cond.release()
        else:
            self.cond.wait()


    def _wait_for_recieve(self, stage):
        '''
        等待梯度，并把接收到的梯度更新到优化器中
        '''
        if self.cond.acquire():
            while not self.is_recieved:
                self.cond.wait()

            # 如果是scatter阶段则累加梯度，同时累加样本数
            if stage == 'scatter':
                self.optimizer.apply_gradients(
                    self.recieved_gradients,  summarize=True, acc_no=self.recieved_acc_no)

            # 如果是all-gather阶段则覆盖梯度，样本数保持不变
            else:
                self.optimizer.apply_gradients(
                    self.recieved_gradients, summarize=False, acc_no=self.optimizer.acc_no)

            self.is_recieved = False

            # 梯度已被更新，通知接收流程继续接收新的梯度
            self.cond.notify_all()
            self.cond.release()
        else:
            self.cond.wait()
