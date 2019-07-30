# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 10:00:02 CST 2019

@author: chenzhen
"""
from core.graph import default_graph


def get_node_from_graph(node_name, name_scope=None, graph=None):
    if graph is None:
        graph = default_graph
    if name_scope:
        node_name = name_scope + '/' + node_name
    for node in graph.nodes:
        if node.name == node_name:
            return node
    return None


def update_node_value_in_graph(node_name, new_value, name_scope=None, graph=None):
    node = get_node_from_graph(node_name, name_scope, graph)
    assert node is not None

    assert node.value.shape == new_value.shape
    node.value = new_value


class name_scope(object):
    def __init__(self, name_scope):
        self.name_scope = name_scope

    def __enter__(self):
        default_graph.name_scope = self.name_scope
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        default_graph.name_scope = None
