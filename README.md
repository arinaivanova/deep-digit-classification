# Simple Neural Network

**Neural network** to **classify MNIST handwritten digits** using machine learning in Python 3. Includes convolutional and fully-connected sigmoid layers. *(intended as an educational project).*

Requires [Numpy](https://scipy.org/install/), a linear algebra library, and the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

### Description

- `main.py`: Train and evaluate network on MNIST dataset using stochastic gradient descent.

<img src="https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png" width=50% height=50%>

  + **input**: set of *28x28* input images *x* and expected outputs *y*.

  + **output**: *10x1* vector of *probabilities* (activations) of the input image *x* corresponding to each digit from *1* to *10*. This is the network's guess, whereas *y* is the correct answer.

- `basic_neuralnet.py`: basic network implementation with fully-connected neurons.

- `neuralnet.py`: modular neural network implementation with convolutional and fully-connected sigmoid layers.
