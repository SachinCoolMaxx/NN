import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

#matplotlib inline

np.random.seed(1)

X,Y =   load_planar_dataset()
#plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);

shape_X = X.shape
shape_Y = Y.shape
m= X.shape[1]
print ('The shape of X is: ' + str(shape_X))
print ('The shape of Y is: ' + str(shape_Y))
print ('I have m = %d training examples!' % (m))

def layer_sizes(X,Y):
	n_x = X.shape[0]
	n_h = 4
	n_y = Y.shape[0]

	return (n_x,n_h,n_y)

X_assess, Y_assess = layer_sizes_test_case()
(n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the hidden layer is: n_h = " + str(n_h))
print("The size of the output layer is: n_y = " + str(n_y))

def initialize_parameters(n_x,n_h,n_y):
	np.random.seed(2)
	W1 = np.random.randn(n_h,n_x) * 0.01
	b1 = np.zeros((n_h,1))
	W2 = np.random.randn(n_y,n_h) * 0.01
	b2 = np.zeros((n_y,1))
	parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

  	return parameters


def forward_propagation(X,parameters):
	W1 = parameters["W1"]
	b1 = parameters["b1"]
 	W2 = parameters["W2"]
 	b2 = parameters["b2"]

 	Z1 = np.dot(W1,X)+b1
 	A1 = np.tanh(Z1)
 	Z2 = np.dot(W2,A1) + b2
 	A2 = sigmoid(Z2)

 	cache = {"Z1": Z1,"A1": A1,"Z2": Z2,"A2": A2}
 	return A2,cache   	


def compute_cost(A2, Y, parameters):
    m = Y.shape[1]
    logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),1-Y)
    cost = (-1/m)*np.sum(logprobs)
    cost = np.squeeze(cost)  
    return cost


def backward_propagation(parameters,cache,X,Y):
	m = X.shape[1]
	W1 = parameters["W1"]
	W2 = parameters["W2"]

	A1 = cache["A1"]
	A2 = cache["A2"]

	dZ2 = A2 - Y
	dW2 = (1/m)*np.dot(dZ2,A1.T)
	db2 = (1/m)*np.sum(dZ2,axis=1,keepdims=True)
	dZ1 = np.multiply(W2.T,dZ2)*(1-np.power(A1,2))
	dW1 = (1/m)*np.dot(dZ1,X.T)
	db1 = (1/m)*np.sum(dZ1,axis=1,keepdims=True)

	grads = {"dW1": dW1,"db1": db1,"dW2": dW2,"db2": db2}
	return grads

def update_parameters(parameters,grads,learning_rate):
	W1 = parameters["W1"]
	W2 = parameters["W2"]
	b1 = parameters["b1"]
	b2 = parameters["b2"]

	dW1 = grads["dW1"]
	dW2 = grads["dW2"]
	db1 = grads["db1"]
	db2 = grads["db1"]

	W1 = W1 - learning_rate*dW1
	W2 = W2 - learning_rate*dW2
	b1 = b1 - learning_rate*db1
	b2 = b2 - learning_rate*db2

  	parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
 	return parameters	