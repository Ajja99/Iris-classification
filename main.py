import numpy as np

def main():
    parameters = initialize_weights([1,2,3])
    print(parameters)

def initialize_weights(layer_dims):

    parameters = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

main()