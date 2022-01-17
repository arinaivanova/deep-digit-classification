"""
Arina Ivanova
16/01/2022

Program to classify MNIST handwritten digits using deep learning
"""

import numpy as np
import random
import mnist_loader, layer, ml_math

""" Train network using stochastic gradient descent """
def sgd(epochs, learning_rate, minibatch_sz, training_set, layers, testing_set=None):
	for epoch in range(epochs):
		# split training set randomly into minibatches
		random.shuffle(training_set)
		minibatches = [training_set[i : i+minibatch_sz] for i in range(0, len(training_set), minibatch_sz)]

		for batch in minibatches:
			for a, y in batch:

				# feed-forward training example through network
				for layer in layers:
					a = layer.forwprop(a)

				# back-propagate cost gradient wrt network output
				grad_a = ml_math.der_mean_quad_cost(a, y)
				for layer in layers[::-1]:
					grad_a = layer.backprop(grad_a, learning_rate)
		
		# evaluate accuracy of network at current training epoch
		if testing_set:
			res = evaluate(testing_set, layers)
			print("epoch: "+str(epoch)+" result: "+str(res)+" / "+str(len(training_set)))

"""
Returns number of correct output activations from network
	
testing_set = (expected output activations, layers of network)
"""
def evaluate(testing_set, layers):
	res = 0
	for a, y in testing_set:
		# feed forward testing example through network
		for layer in layers:
			a = layer.forwprop(a)
		# index of neuron in last layer with highest activation
		predicted_digit = np.argmax(a)
		if predicted_digit == y:
			res += 1
	return res

# init network

layers = [
	layer.Dense(30, 784),
	layer.Activation(ml_math.sigmoid, ml_math.der_sigmoid),
	layer.Dense(10, 30),
	layer.Activation(ml_math.sigmoid, ml_math.der_sigmoid)
]

# load MNIST set

training_set, validation_set, testing_set = mnist_loader.load_data("data/mnist.pkl.gz")

training_set = [ (x,y) for x,y in zip(training_set[0],training_set[1]) ]
testing_set = [ (x,y) for x,y in zip(testing_set[0],testing_set[1]) ]

# set hyperparameters

epochs = 100
learning_rate = 3.0
minibatch_sz = 30

# train fully-connected neural net on MNIST dataset
sgd(epochs, learning_rate, minibatch_sz, training_set, layers, testing_set)