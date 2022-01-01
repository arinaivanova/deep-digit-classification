"""
Arina Ivanova
https://github.com/arinaivanova/simple-neural-net

Program to classify MNIST handwritten digits using neural network. Includes convolutional and fully connected layers.
"""
import numpy as np
import mnist_loader
import neuralnet as nn

# train fully-connected neural net on MNIST dataset

training_set, validation_set, testing_set = mnist_loader.load_data("../data/mnist.pkl.gz")

net = nn.Network([
    nn.FCLayer(100, 784),
    nn.FCLayer(10, 100)
    ])

training_set = [ (x,y) for x,y in zip(training_set[0],training_set[1]) ]
testing_set = [ (x,y) for x,y in zip(testing_set[0],testing_set[1]) ]

"""
# Convolutional neural net

width = 28
depth = 1
k = 5
net = nn.Network([
    nn.ConvLayer(k, depth, width),
    nn.FCLayer(10, depth)
    ])

image_set = [np.zeros(shape=(width,width))]*len(training_set[0])
for n in range(0,len(training_set[0])):
    for i in range(0,width):
        for j in range(0,width):
            image_set[n][i][j] = training_set[0][i*width+j][j]

training_set = [(x,y) for x,y in zip(image_set, training_set[1])]
"""

epochs = 50
learning_rate = 0.01
sample_sz = 30

# train network using stochastic gradient descent, and evaluate
net.sgd(training_set,epochs,sample_sz,learning_rate,testing_set=testing_set)

