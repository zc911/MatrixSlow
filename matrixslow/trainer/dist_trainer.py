# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 20:48:16 CST 2019

@author: chenzhen
"""
from . import trainer
from dist import ps


class SyncTrainerParameterServer(trainer.Trainer):
    def __init__(self, *args, **kargs):
        trainer.Trainer.__init__(self, *args, **kargs)
        cluster_conf = kargs['cluster_conf']
        ps_host = cluster_conf['ps'][0]
        self.ps_client = ps.ParameterServiceClient(ps_host)

    def _optimizer_update(self):
        acc_gradient = self.optimizer.acc_gradient
        self.ps_client.push_gradients(
            acc_gradient, self.optimizer.acc_no)
        nodes_name = [node.name for node in acc_gradient.keys()]
        node_gradients_dict = self.ps_client.pull_gradients(
            nodes_name)
        self.optimizer.update(node_gradients_dict)
