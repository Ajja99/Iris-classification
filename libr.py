import numpy as np
import activations

def initialize_parameters(layer_dims):

    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

def linear_forward(A, W, b):

    Z = np.add(np.dot(W, A), b)
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):

    Z, Z_cache = linear_forward(A_prev, W, b)

    if activation == 'relu':
        A, A_cache = activations.relu(Z)

    elif activation == 'sigmoid':
        A, A_cache = activations.sigmoid(Z)

    elif activation == 'softmax':
        A_cache = activations.softmax(Z)

    cache = (Z_cache, A_cache)

    return A, cache

def model_forward(X, parameters):

    A = X
    caches = []
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')

        caches.append(cache)
    
    aL, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'softmax')
    caches.append(cache)

    return aL, caches


def compute_cost(aL, Y):
    m = len(Y)

    print(aL)
    print(Y)

    loss = -np.sum(Y*np.log(aL))

    cost = -(1/m) * np.sum(loss)

    return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dZ, A_prev)
    db = (1/m) * np.sum(dZ)
    dA_prev = np.dot(W.t, dZ)

    return dA_prev, W, b

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == 'relu':
        dZ = np.dot(dA, activations.relu_backward(dA, activation_cache))
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == 'sigmoid':
        dZ = np.dot(dA, activations.sigmoid_backward(dA, activation_cache))
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def model_backward(aL, Y, caches):

    grads = {}
    L = len(caches)
    m = aL.shape[1]
    Y = Y.reshape(aL.shape)

    daL = aL-Y

    current_cache = caches[L]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = daL

    for l in reversed(range(L-1)):
        current_cache = caches[l]

        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, 'relu')

        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

        return grads

def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l+1)] -= np.multiply(learning_rate, grads['dW'+ str(l)])
        parameters["b" + str(l+1)] -= np.multiply(learning_rate, grads['db'+ str(l)])

    return parameters