# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 14:53:57 CST 2019

@author: chenzhen
"""
import matrixslow as ms


class Exporter(object):
    '''
    模型服务导出
    '''
    def __init__(self, graph=None):
        self.graph = ms.default_graph if graph is None else graph

    def signature(self, input_name, output_name):
        '''
        返回模型服务接口签名。通过确定输入和输出节点的名称，作为模型服务接口的最终
        输入和输出，从而保存到模型文件中
        '''
        input_var = ms.get_node_from_graph(input_name, graph=self.graph)
        assert input_var is not None
        output_var = ms.get_node_from_graph(output_name, graph=self.graph)
        assert output_var is not None

        input_sigature = dict()
        input_sigature['name'] = input_var.name

        output_signature = dict()
        output_signature['name'] = output_var.name
        return {
            'inputs': input_sigature,
            'outputs': output_signature
        }
