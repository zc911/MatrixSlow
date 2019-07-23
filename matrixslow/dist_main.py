import sys

import grpc
from dist import ps
from dist.proto import parameter_server_pb2 as pspb
from dist.proto import parameter_server_pb2_grpc as psrpc

sys.path.append('.')


def client():

    nodes = []
    gradients = []
    node_gradients = pspb.NodeGradients(nodes=nodes, gradients=gradients)
    push_req = pspb.ParameterPushReq(token=1, node_gradients=node_gradients)
    response = stub.Push(push_req)
    print("client received: ", response)
    # response = stub.Pull(helloworld_pb2.HelloRequest(name='daydaygo'))
    # print("client received: " + response)


if __name__ == '__main__':
    role = sys.argv[1]
    if role == 'ps':
        ps.serve()
    else:
        client()
    line_count = {}
