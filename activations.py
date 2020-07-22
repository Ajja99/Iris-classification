import numpy as np

def sigmoid(Z):
    return 1/(1+np.exp(-Z)), Z

def relu(Z):
    return max(0, Z), Z

def softmax(Z):
    return np.exp(Z)/(np.sum(np.exp(Z))), Z

def sigmoid_backward(dA, cache):
    Z = cache
    
    s = sigmoid(Z)
    dZ = dA * s * (1-s)

    return dZ

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)

    dZ[Z <= 0] = 0

    return dZ

def softmax_backward(aL, Y):

    dZ = np.subtract(aL, Y)

    return dZ

    