import libr as lib
import sklearn.datasets
import matplotlib.pyplot as plt
import numpy as np



def load_data():
    iris_dataset = sklearn.datasets.load_iris()
    X, Y = iris_dataset['data'], iris_dataset['target']

    # Yoh = np.zeros((Y.max()+1, Y.size))
    # Yoh[np.arange(Y.size),Y] = 1

    print(Yoh)

def model(X, Y, layers_dims, learning_rate, num_iterations, print_cost):
    costs = []

    parameters = lib.initialize_parameters(layers_dims)

    for i in range(0, num_iterations):

        aL, caches = lib.model_forward(X, parameters)

        cost = lib.compute_cost(aL, Y)

        grads = lib.model_backward(aL, Y, caches)

        parameters = lib.update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

load_data()