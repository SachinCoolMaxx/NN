import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

%matplotlib inline
np.random.seed(1)

#initialzing a variable in tensor flow
y_hat = tf.constant(36,name='y_hat')
y = tf.constant(39,name = 'y')
loss  =  tf.variable((y-y_hat)**2,name = 'loss')
init = tf.global_variables_initializer()
 
 with tf.Session() as session:
 	session.run(init)
 	print(session.run(loss))

#another way to do teh same 
a = tf.constant(2)
b = tf.constant(10)
c = tf.multiply(a,b)
print(c)
sess = tf.Session()
print(sess.run(c))

#how to specify a place holder 
x = tf.placeholder(tf.init64,name = 'x')
print(sess.run(2*x,feed_dict = {x:3}))
sess.close()

# How to implement a sigmoid in tf
def sigmoid(z):
	x = tf.placeholder(tf.float32,name-'x')
	sigmoid = tf.sigmoid(x)
	with tf.Session as sess:
		result = sess.run(sigmoid,feed_dict={x:z})
	return result

print ("sigmoid(0) = " + str(sigmoid(0)))


#Tensor flow help us to compute the cost function very easily using the built in sigmoid_cross_entropy funtion
def cost(logits,labels):
	z = tf.placeholder(tf.float32,name='z')
	y = tf.placeholder(tf.float32,name='y')
	cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = z,labels = y)

	sess = tf.Session()
	cost = sess.run(cost,feed_dict={z:logits,y:labels})

	return cost

logits = sigmoid(np.array([0.2,0.4,0.7,0.9]))
cost = cost(logits, np.array([0,0,1,1]))
print ("cost = " + str(cost))


## Implement a 3 Layer Naeural Network to classify the imaegs af
#create a placehilder for the inputs X and the putput Y
#we donnot know the number of exaples m , so we will use None in plcae of theat

def create_placeholders(n_x,n_y):
	X = tf.placeholder(tf.float32,shape=(n_x,None),name='X')
	Y = tf.placeholder(tf.float32,shape=(n_y,None),name='y')

	return X,Y

def initialize_parameters():
	#we will use Xavier's initailxation method to initiale the various paramters
	W1 = tf.get_variable("W1",[25,12288],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1",[25,1],initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2",[12,25],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2",[12,1],initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3",[6,12],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3",[6,1],initializer = tf.zeros_initializer())

    parameters = {"W1": W1,"b1": b1,"W2": W2,"b2": b2,"W3": W3,"b3": b3}

    return parameters

def forward_propagation(X,parameters):
	W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    #u dont need to compute A3, becz the built in loss function for the tensor flow automaticaally computes A3, logiys are Z3

    Z1 = tf.add(tf.matmul(W1,X),b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1),b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2),b3)

	return Z3 

#It is important to know that the "logits" and "labels" inputs of tf.nn.softmax_cross_entropy_with_logits are expected to be of shape (number of examples, num_classes). We have thus transposed Z3 and Y for you.

def compute_cost(Z3,Y):
	logits = tf.transpose(Z3)
	labels = tf.transpose(Y)

	cost = cost(logits,labels)

	return cost



#The whole model

def model(X_train,Y_train,X_test,Y_test,learning_rate=0.0001,num_epochs=1500,minibatch_size=32,print_cost):
	ops.reset_default_graph()
	tf.set_random_seed(1)
	seed=3
	(n_x,m) = X_train.shape
	n_y = Y_train.shape[0]
	costs = []

	X,Y = create_placeholders(n_x,n_y)

	parameters = initialize_parameters()

	Z3 = forward_propagation(X,parameters)

	cost = compute_cost(Z3,Y)

	optimizer = tf.train.AdamOptimizer(learning_rate =learning_rate).minimize(cost)
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)

		for epoch in range(num_epochs):
			epoch_cost=0
			num_minibatches = int(m/minibatch_size)
			seed=seed+1;
			minibatches = random_mini_batches(X_train,Y_train,minibatch_size,seed)

			for minibatch in minibatches:
				(minibatch_X,minibatch_Y) = minibatch

				_,minibatch_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X,Y:minibatch_Y})

				epoch_cost += minibatch_cost/num_minibatches


            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)	

         # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        parameters = sess.run(parameters)

 	return parameters