import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases_v3 import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

def initialize_parameters(n_x,n_h,n_y):
	np.random.seed(1)
	W1 = np.random.randn(n_h,n_x) * 0.01
	W2 = np.random.randn(n_y,n_h) * 0.01
	b1 = np.random.randn((n_h,1)) * 0.01
	b2 = np.random.randn((n_y,1)) * 0.01

  parameters = {"W1": W1,"b1": b1,"W2": W2,"b2": b2}
  return parameters

def initialize_parameters_deep(layer_dims):
	np.random.seed(3)
	parameters{}
	L= len(layer_dims)

	for l in range(1,L)
	parameters["W" + str(l) ] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
	parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

	return parameters

def linear_forward(A,W,b):
	Z = np.dot(W,A) +b
	cache = (A,W,b)

	return Z,cache

def linear_activation_forward(A_prev,W,b,activation):
	if activation == "sigmoid":
		Z,linear_cache = linear_forward(A_prev,W,b)
		A,activation_cache = sigmoid(Z)

	elif activation == "relu":
		Z,linear_cache = linear_forward(A_prev,W,b)
		A,activation_cache = relu(Z)

	cache = (linear_cache,activation_cache)
	return A,cache

def L_model_forward(X,parameters):
	caches = []
	A=X
	L = len(parameters)

	for l in range(1,L):
		A_prev = A
		A,cache = linear_activation_forward(A_prev,parameters['W' + str(l)],parameters['b' + str(l)],"relu")
		caches.append(cache)

	A_prev = A
	AL,cache = linear_activation_forward(A_prev,parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
	caches.append(cache)

	return AL,caches

def compute_cost(AL,Y):
	m = Y.shape[1]
	cost = (-1/m)*np.sum(np.multiply(np.log(AL),Y) + np.multiply(np.log(1-AL),1-Y))
	cost = np.squeeze(cost)
	return cost

def linear_backward(dZ,cache):
	A_prev, W,b = cache
	m  = A_prev.shape[1]

	dW = (1/m)*np.dot(dZ,A_prev.T)
	db = (1/m)*np.sum(dZ,axis=1,keepdims=True)
	dA_prev = np.dot(W.T,dZ)

	return dA_prev,dW,db

def linear_activation_backward(dA,cache,activation):
	linear_cache,activation_cache = cache

	if activation == "relu":
		dZ = relu_backward(dA,activation_cache)
		dA_prev,dW,db = linear_backward(dZ,linear_cache)

	elif activation == "sigmoid":
		dZ = sigmoid_backward(dA,activation_cache)
		dA_prev, dW, db = linear_backward(dZ,linear_cache)

	return dA_prev,dW,db


def L_model_backward(AL,Y,caches):
	grads = {}
	L = len(caches)
	m = AL.shape[1]
	Y = Y.reshape(AL.shape)

	dAL = -np.divide(Y, AL) + np.divide(1 - Y, 1 - AL)

	#going back from the last layer of sigmoid to linear
	current_cache = caches[L-1] 	#remember that the counting of caches ( a list) has started from 0 , otherwise for backprpagatio of layer L , u will use the cache of same number only
	grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")	

	for l in (reversed(range(L))):
		if l>0:
			current_cache = caches[l-1]
			dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, "relu")
			grads["dA" + str(l )] = dA_prev_temp
			grads["dW" + str(l )] = dW_temp			
			grads["db" + str(l )] = db_temp

	return grads


def update_parameters(parameters,grads,learning_rate):
	L - len(parameters)

	for i in range(1,L+1):
		parameters["W" + str(i)] = parameters["W" + str(i)] - learning_rate*grads["dW" + str(i)]
		parameters["b" + str(i)] = parameters["b" + str(i)] - learning_rate*grads["db" + str(i)]
	
	return parameters		