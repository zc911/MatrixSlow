# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 15:55:34 CST 2019

@author: chenzhen
"""
import json
import os
import datetime

import numpy as np


from ..core.core import get_node_from_graph
from ..core import *
from ..core import Node, Variable
from ..core.graph import default_graph
from ..ops import *
from ..ops.loss import *
from ..ops.metrics import *
from ..util import ClassMining


class Saver(object):
    '''
    模型、计算图保存和加载工具类
    模型保存为两个单独的文件：
    1. 计算图自身的结构元信息
    2. 节点的值，更确切的说是变量节点的权值
    '''

    def __init__(self, root_dir=''):
        self.root_dir = root_dir
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)


    def save(self, graph=None, meta=None, service_signature=None,
             model_file_name='model.json',
             weights_file_name='weights.npz'):
        '''
        把计算图保存到文件中
        '''

        if graph is None:
            graph = default_graph

        # 元信息，主要记录模型的保存时间和节点值文件名
        meta = {} if meta is None else meta
        meta['save_time'] = str(datetime.datetime.now())
        meta['weights_file_name'] = weights_file_name

        # 服务接口描述
        service = {} if service_signature is None else service_signature

        # 开始保存操作
        self._save_model_and_weights(
            graph, meta, service, model_file_name, weights_file_name)


    def _save_model_and_weights(self, graph, meta, service, model_file_name, weights_file_name):
        model_json = {
            'meta': meta,
            'service': service
        }
        graph_json = []
        weights_dict = dict()

        # 把节点元信息保存为dict/json格式
        for node in graph.nodes:
            if not node.need_save:
                continue
            node.kargs.pop('name', None)
            node_json = {
                'node_type': node.__class__.__name__,
                'name': node.name,
                'parents': [parent.name for parent in node.parents],
                'children': [child.name for child in node.children],
                'kargs': node.kargs
            }

            # 保存节点的dim信息
            if node.value is not None:
                if isinstance(node.value, np.matrix):
                    node_json['dim'] = node.value.shape

            graph_json.append(node_json)

            # 如果节点是Variable类型，保存其值
            # 其他类型的节点不需要保存
            if isinstance(node, Variable):
                weights_dict[node.name] = node.value

        model_json['graph'] = graph_json

        # json格式保存计算图元信息
        model_file_path = os.path.join(self.root_dir, model_file_name)
        with open(model_file_path, 'w') as model_file:
            json.dump(model_json, model_file, indent=4)
            print('Save model into file: {}'.format(model_file.name))

        # npz格式保存节点值（Variable节点）
        weights_file_path = os.path.join(self.root_dir, weights_file_name)
        with open(weights_file_path, 'wb') as weights_file:
            np.savez(weights_file, **weights_dict)
            print('Save weights to file: {}'.format(weights_file.name))


    @staticmethod
    def create_node(graph, from_model_json, node_json):
        '''
        静态工具函数，递归创建不存在的节点
        '''
        node_type = node_json['node_type']
        node_name = node_json['name']
        parents_name = node_json['parents']
        dim = node_json.get('dim', None)
        kargs = node_json.get('kargs', None)
        kargs['graph'] = graph

        parents = []
        for parent_name in parents_name:
            parent_node = get_node_from_graph(parent_name, graph=graph)
            if parent_node is None:
                parent_node_json = None
                for node in from_model_json:
                    if node['name'] == parent_name:
                        parent_node_json = node

                assert parent_node_json is not None
                # 如果父节点不存在，递归调用
                parent_node = Saver.create_node(
                    graph, from_model_json, parent_node_json)

            parents.append(parent_node)

        # 反射创建节点实例
        if node_type == 'Variable':
            assert dim is not None

            dim = tuple(dim)
            return ClassMining.get_instance_by_subclass_name(Node, node_type)(*parents, dim=dim, name=node_name, **kargs)
        else:
            return ClassMining.get_instance_by_subclass_name(Node, node_type)(*parents, name=node_name, **kargs)


    def _restore_nodes(self, graph, from_model_json, from_weights_dict):

        for index in range(len(from_model_json)):
            node_json = from_model_json[index]
            node_name = node_json['name']

            weights = None
            if node_name in from_weights_dict:
                weights = from_weights_dict[node_name]

            # 判断是否创建了当前节点，如果已存在，更新其权值
            # 否则，创建节点
            target_node = get_node_from_graph(node_name, graph=graph)
            if target_node is None:
                print('Target node {} of type {} not exists, try to create the instance'.format(
                    node_json['name'], node_json['node_type']))
                target_node = Saver.create_node(
                    graph, from_model_json, node_json)

            target_node.value = weights


    def load(self, to_graph=None,
             model_file_name='model.json',
             weights_file_name='weights.npz'):
        '''
        从文件中读取并恢复计算图结构和相应的值
        '''
        if to_graph is None:
            to_graph = default_graph

        model_json = {}
        graph_json = []
        weights_dict = dict()

        # 读取计算图结构元数据
        model_file_path = os.path.join(self.root_dir, model_file_name)
        with open(model_file_path, 'r') as model_file:
            model_json = json.load(model_file)

        # 读取计算图节点值数据
        weights_file_path = os.path.join(self.root_dir, weights_file_name)
        with open(weights_file_path, 'rb') as weights_file:
            weights_npz_files = np.load(weights_file)
            for file_name in weights_npz_files.files:
                weights_dict[file_name] = weights_npz_files[file_name]
            weights_npz_files.close()

        graph_json = model_json['graph']
        self._restore_nodes(to_graph, graph_json, weights_dict)
        print('Load and restore model from {} and {}'.format(
            model_file_path, weights_file_path))

        self.meta = model_json.get('meta', None)
        self.service = model_json.get('service', None)
        return self.meta, self.service
