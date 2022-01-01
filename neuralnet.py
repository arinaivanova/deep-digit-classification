"""
Arina Ivanova
https://github.com/arinaivanova/simple-neural-net

layers, neural net and utility functions. Includes convolutional and fully-connected layers.
"""
import numpy as np
import random

class Layer:

    def feedforw():
        pass

    def backprop():
        pass

class FCLayer(Layer):

    # n,m = (num. neurons in cur layer, num. neurons in prev. layer)
    def __init__(self, n, m):

        self.biases = np.random.normal(0, 1, size=(n, 1))
        self.weights = np.random.normal(0, 1, size=(n, m))

    def feedforw(self, x):

        z = np.dot(self.weights, x) + self.biases
        a = sigmoid(z)

        return a, z

class ConvLayer(Layer):

    # k = filter size
    # depth = number of feature maps
    # width = width of square image
    def __init__(self, k, depth, width):

        self.kernel_sz = k
        self.depth = depth
        self.width = width
        self.biases = np.random.normal(0, size=(depth,1) )
        self.weights = np.random.normal(0, 1, size=(depth,k,k)) 

    def feedforw(self, inp):
        
        z = np.zeros(self.biases.shape)
        a = np.zeros(self.biases.shape)

        for y in range(0, inp.shape[0] - self.kernel_sz + 1):
            for x in range(0, inp.shape[1] - self.kernel_sz + 1):

                # KxK sub-region of input image
                r = inp[y:(y+self.kernel_sz), x:(x+self.kernel_sz)]
                for j in range(0,self.depth):
                    z[j] = self.biases[j] + conv2d(self.weights[j], r)
                    a[j] = sigmoid(z[j])

        return a, z


class Network:

    def __init__(self, layers):

        self.num_layers = len(layers)+1 # include input layer in size
        self.layers = layers

    # returns lists of sigmoid and weighted input vectors for network given sigmoids 'x' of input layer
    # x = sigmoid vector of input layer
    def feedforward(self, x):
        a = [x]
        z = []
        # update a for weight matrix w and bias vector b in each layer N (excluding input layer)
        for l in self.layers:

            a_l, z_l = l.feedforw(a[-1])
            z.append(z_l)
            a.append(a_l)

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
        grads_w = [np.zeros(l.weights.shape) for l in self.layers]
        grads_b = [np.zeros(l.biases.shape) for l in self.layers]

        # x,y = i-th training ex
        for x,y in sample:
            
            a, z = self.feedforward(x)
            # vectors of change in cost gradients wrt weights and biases, resp. provdided new training ex. x_i
            # i-th = change in gradient of cost(training example x_i) wrt w and b, resp.
            d_grads_w, d_grads_b = self.backprop(a,z,x,y)

            # apply Newton's method to compute next step of descent for each layer
            # gives improved approx of minimal weights and biases, resp, of each layer, provided new training ex. x_i
            for n in range(0, self.num_layers - 1):

                # update each gradient of C_x_i for each layer with its corresponding change d_grad (from backprop.)
                grads_w[n] = grads_w[n] + d_grads_w[n]
                grads_b[n] = grads_b[n] + d_grads_b[n]

                self.layers[n].weights = self.layers[n].weights - rate/len(sample)*grads_w[n]
                self.layers[n].biases = self.layers[n].biases - rate/len(sample)*grads_b[n]

    # a[i] = sigmoid vector of layer i
    # z[i] = weighted input vector of layer i
    # (x,y) = training example: Lx1 input and expected output vector
    # returns lists (vector for each layer) of change in gradients of cost of training example x wrt each weight and bias, resp.
    def backprop(self,a,z,x,y):

        # Nx1 vector
        delta = der_sigmoid(z[-1]) * der_cost(a[-1],y)

        der_cx_w = [ np.zeros(l.weights.shape) for l in self.layers ]
        der_cx_b = [ np.zeros(l.biases.shape) for l in self.layers ]

        # der_cx_w[L] = NxM matrix
        der_cx_w[-1] = np.matmul(delta, a[-2].transpose())
        # der_cx_b[L] = Nx1 vector
        der_cx_b[-1] = delta

        for l in range(2, self.num_layers - 1):

            # (((NxM)T->(MxN) dot Nx1)->Mx1 * Mx1)->Mx1 vector
            delta = np.matmul(self.layers[-l+1].weights, delta) * der_sigmoid(z[-l])
            # (Mx1 dot (Px1)T->1xP)->MxP matrix
            der_cx_w[-l] = np.matmul(delta, a[-l-1].transpose())
            # Mx1 vector
            der_cx_b[-l] = delta

        return (der_cx_w,der_cx_b)

    # return number of correct sigmoids
    def evaluate(self,training_set):

        res = []
        for x,y in training_set:
            # index of neuron in last layer with highest sigmoid
            predicted_y = np.argmax(self.feedforward(x)[0][-1])
            res.append( (predicted_y, y) )

        return sum(int(a == b) for a,b in res)

# misc functions

def sigmoid(z):

    return 1.0 / (1.0 + np.exp(-z))

def der_sigmoid(z):

    pow = np.exp(z)
    return 1/(pow + 1/pow + 2)

def der_cost(a,y):

    return 2*(a - y)

def conv2d(weights, r):

    res = 0
    for y in range(0,len(weights)):
        for x in range(0,len(weights)):
            res += weights[y][x] + r[y][x]

    return res

