#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: chenzhen
@Date: 2020-04-22 14:57:10
@LastEditTime: 2020-04-22 15:29:33
@LastEditors: chenzhen
@Description:
'''
import sys
sys.path.append('../../')
import matrixslow_serving as mss


print(mss.serving.MatrixSlowServer)

serving = mss.serving.MatrixSlowServer(
    host='127.0.0.1:5000', root_dir='./epoches10', model_file_name='my_model.json', weights_file_name='my_weights.npz')

serving.serve()
