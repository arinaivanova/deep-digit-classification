"""
Arina Ivanova
15/01/2022
"""

import numpy as np

class Layer:

	def __init__(self):
		pass

	def forwprop(self):
		pass

	def backprop(self):
		pass

class Dense(Layer):

	"""	
	Randomly initialize weights and biases of current layer

	n,m = (num. neurons in cur layer, num. neurons in prev. layer)
	"""
	def __init__(self, n, m):
		self.w = np.random.normal(0, 1, size=(n, m))
		self.b = np.random.normal(0, 1, size=(n, 1))
	
	"""
	Apply weights and biases to input; return layer's activations

	x = inputs to current layer
	"""
	def forwprop(self, x):
		self.x = x
		a = np.dot(self.w, x) + self.b
		return a

	"""
	Apply gradient descent to current layer; returns cost gradients wrt its input

	grad_a = cost gradients wrt activations of current layer
	rate = learning rate of gradient descent 
	"""
	def backprop(self, grad_a, rate):
		# compute cost gradients wrt weights and biases
		grad_w = np.dot(grad_a, self.x)
		grad_b = grad_a
		# improve precision of weights and biases w/ gradient descent
		self.w -= rate*grad_w
		self.b -= rate*grad_b
		# return cost gradients wrt input activations
		grad_x = np.dot(self.w.transpose(), self.x)
		return grad_x

class Activation(Layer):

	def __init__(self, activation, der_activation):
		self.activation = activation
		self.der_activation = der_activation

	def forwprop(self, x):
		self.x = x
		a = self.activation(x)
		return a
	
	def backprop(self, grad_a):
		grad_x = self.der_activation(self.x) * grad_a
		return grad_x
