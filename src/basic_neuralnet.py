"""
Arina Ivanova
https://github.com/arinaivanova/ml-digit-classification

artificial neural network and utility functions to classify MNIST handwritten digits using machine learning
"""

import numpy as np
import random

# misc functions

def activation(z):

	return 1.0 / (1.0 + np.exp(-z))

def der_activation(z):

	pow = np.exp(z)
	return 1/(pow + 1/pow + 2)

def der_cost(a,y):

	return 2*(a - y)

# represents neural network
class Network:

	# constructor
	# sizes = num. neurons in each layer
	def __init__(self, sizes):

		self.num_layers = len(sizes)
		self.sizes = sizes

		# list of (num. layers - 1) vecs with i-th = biases of i-th layer (excluding input layer)
		# randomly generated from normal distribution with mean=0, standard deviation=1
		self.biases = [ np.random.normal(0, 1, size=(n, 1)) for n in sizes[1:] ]

		# list of (num. layers - 1) vecs with i-th = weights of i-th layer (excluding input layer)
		# self.weights[j][k] = weight of edge between k-th neuron in (N-1)-st layer and j-th in N-th
		self.weights = [ np.random.normal(0, 1, size=(cur, prev)) for prev,cur in zip(sizes[:-1], sizes[1:]) ]

	# returns lists of activation and weighted input vectors for network given activations 'x' of input layer
	# x = activation vector of input layer
	def feedforward(self, x):

		a = [x]
		z = []
		# update a for weight matrix w and bias vector b in each layer N (excluding input layer)
		for w,b in zip(self.weights,self.biases):

			# apply weights and biases of j-th neuron of N-th layer to activation of each neuron k in (N-1)st layer
			# sigmoid applied element-wise to resultant vector

			z.append(np.dot(w, a[-1]) + b)
			a.append(activation(z[-1]))

		return a, z

	# training_set = list of training examples (x,y) where x=input and y=expected output
	# testing_set = "       " testing "                                                "
	# epochs = num. of training epochs
	# sample_sz = size of random samples from training data
	# rate = learning rate
	def sgd(self, training_set, epochs, sample_sz, rate, testing_set=None):

		for e in range(epochs):

			# randomize order of training examples
			random.shuffle(training_set)
			# divide training data into samples
			samples = [training_set[i:(i+sample_sz)] for i in range(0, len(training_set), sample_sz)]

			for x in samples:
				self.update_sample(x, rate)

			if testing_set:
				print("epoch: "+str(e)+" result: "+str(self.evaluate(testing_set))+" / "+str(len(testing_set)))

	# improve weights and biases of all neurons by taking a step of descent with avg. over all sample
	# sample = list of (x,y) sampled from training data where x=input and y=desired output
	def update_sample(self, sample, rate):

		# initialize gradients of C_x_i (for all layers)
		# i-th in list correspond to i-th layer (excluding input layer)
		grads_w = [np.zeros(w.shape) for w in self.weights]
		grads_b = [np.zeros(b.shape) for b in self.biases]

		# x,y = i-th training ex
		for x,y in sample:
			
			a, z = self.feedforward(x)

			# vectors of change in cost gradients wrt weights and biases, resp. provdided new training ex. x_i
			# i-th = change in gradient of cost(training example x_i) wrt w and b, resp.
			d_grads_w, d_grads_b = self.backprop(a,z,x,y)

			# update each gradient of C_x_i for each layer with its corresponding change d_grad (from backprop.)
			grads_w = [grad_w + d_grad_w for grad_w,d_grad_w in zip(grads_w,d_grads_w)]
			grads_b = [grad_b + d_grad_b for grad_b,d_grad_b in zip(grads_b,d_grads_b)]

			# apply Newton's method to compute next step of descent for each layer
			# gives improved approx of minimal weights and biases, resp, of each layer, provided new training ex. x_i
			self.weights = [ w - rate/len(sample)*g for w,g in zip(self.weights,grads_w) ]
			self.biases = [ b - rate/len(sample)*g for b,g in zip(self.biases,grads_b) ]

	# a[i] = activation vector of layer i
	# z[i] = weighted input vector of layer i
	# (x,y) = training example: Lx1 input and expected output vector
	# returns lists (vector for each layer) of change in gradients of cost of training example x wrt each weight and bias, resp.
	def backprop(self,a,z,x,y):

	  	# Nx1 vector
		delta = der_activation(z[-1]) * der_cost(a[-1],y)

		der_cx_w = [ np.zeros(w.shape) for w in self.weights ]
		der_cx_b = [ np.zeros(b.shape) for b in self.biases ]

		# der_cx_w[L] = NxM matrix
		der_cx_w[-1] = np.matmul(delta, a[-2].transpose())
		# der_cx_b[L] = Nx1 vector
		der_cx_b[-1] = delta

		for l in range(2, self.num_layers - 1):

			# (((NxM)T->(MxN) dot Nx1)->Mx1 * Mx1)->Mx1 vector
			delta = np.matmul(self.weights[-l+1], delta) * der_activation(z[-l])
			# (Mx1 dot (Px1)T->1xP)->MxP matrix
			der_cx_w[-l] = np.matmul(delta, a[-l-1].transpose())
			# Mx1 vector
			der_cx_b[-l] = delta

		return (der_cx_w,der_cx_b)

	# return number of correct activations
	def evaluate(self,training_set):

		res = []
		for x,y in training_set:
			# index of neuron in last layer with highest activation
			predicted_y = np.argmax(self.feedforward(x)[0][-1])
			res.append( (predicted_y, y) )

		return sum(int(a == b) for a,b in res)
