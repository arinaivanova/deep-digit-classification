"""
Arina Ivanova
https://github.com/arinaivanova/ml-digit-classification

Program to classify MNIST handwritten digits using machine learning with artificial neural network
"""

import numpy as np
import mnist_loader
import neuralnet

# MNIST dataset

net = neuralnet.Network([784, 100, 10])
print(net.num_layers )
print(net.sizes      )

training_set, validation_set, testing_set = mnist_loader.load_data("../data/mnist.pkl.gz")

training_set = [ (x,y) for x,y in zip(training_set[0],training_set[1]) ]
testing_set = [ (x,y) for x,y in zip(testing_set[0],testing_set[1]) ]

epochs = 15
learning_rate = 0.01
sample_sz = 30

net.sgd(training_set,epochs,sample_sz,learning_rate,testing_set=testing_set)
