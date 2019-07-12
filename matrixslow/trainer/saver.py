# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 15:55:34 CST 2019

@author: chenzhen
"""

import pickle

from core.graph import default_graph


class Saver(object):
    def save(self, graph=None):
        if graph:
            pass
        print('save to model.plk')
        print(default_graph.nodes)
        savefile = open('./model.pkl', 'w')
        pickle.dump(default_graph, savefile)
        savefile.close()

    def load(self, model=None, weights=None, to_graph=None):
        print('load from model.pkl')
        graph = pickle.load('./model.plk')
        print(graph.nodes)
