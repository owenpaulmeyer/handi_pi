import numpy as np

class Network(object):

    def __init__(self, sizes):
        # sizes := a list what contains the number of neurons per respective layer. ie: 3 layers with n neurons: [2, 5, 3]
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # if W := weights of layer n ( self.weights[n] ), then:
        # Wjk ( self.weights[n][j][k] ) is the weight for the connection between the kth neuron in the nth layer and the jth neuron in the (n+1)th layer
        self.weights = [np.random.randn(y,x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
