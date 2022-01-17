"""
Arina Ivanova
15/01/2022
"""

import numpy as np

""" Cost and activation functions and their derivatives """

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def der_sigmoid(z):
    pow = np.exp(z)
    return 1/(pow + 1/pow + 2)

def mean_quad_cost(a, y):
	return (a - y)**2/2/len(a)

def der_mean_quad_cost(a, y):
	return (a - y)/len(a)