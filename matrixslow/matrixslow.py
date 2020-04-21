#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: chenzhen
@Date: 2020-04-03 19:40:23
@LastEditTime: 2020-04-20 17:00:45
@LastEditors: chenzhen
@Description:
'''
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 10:00:02 CST 2019

@author: chenzhen
"""


from . import core
# from . import dist
from . import layer
from . import ops
from . import optimizer
from . import trainer

default_graph = core.default_graph
get_node_from_graph = core.get_node_from_graph
name_scope = core.name_scope
Variable = core.Variable
