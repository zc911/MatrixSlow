

import sys
sys.path.append('.')
sys.path.append('../')
import matrixslow_serving as mss

if __name__ == '__main__':
    host = 'localhost:5000'
    root_dir = './export'
    model_file_name = 'my_model.json'
    weights_file_name = 'my_weights.npz'
    serving = mss.serving.MatrixSlowServing(
        host, root_dir, model_file_name, weights_file_name)
    serving.serve()
