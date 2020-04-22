#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: chenzhen
@Date: 2019-08-28 10:25:40
@LastEditTime: 2020-04-22 15:07:14
@LastEditors: chenzhen
@Description:
'''


import matrixslow_serving as mss
import sys
sys.path.append('.')
sys.path.append('../')

if __name__ == '__main__':
    host = 'localhost:5000'
    root_dir = './export'
    model_file_name = 'my_model.json'
    weights_file_name = 'my_weights.npz'
    serving = mss.serving.MatrixSlowServer(
        host, root_dir, model_file_name, weights_file_name)
    serving.serve()
