import numpy as np

class Network(object):

    def __init__(self, sizes):
        # sizes := a list what contains the number of neurons per respective layer. ie: 3 layers with n neurons: [2, 5, 3]
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
