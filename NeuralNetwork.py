import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

class DenseLayer:
    def __init__(self, ninputs, nneurons, wtregularizer1=0, wtregularizer2=0, bsregularizer1=0, bsregularizer2=0):
        self.weights = 0.01 * np.random.randn(ninputs, nneurons)
        self.biases = np.zeros((1, nneurons))
        self.wtregularizer1 = wtregularizer1
        self.wtregularizer2 = wtregularizer2
        self.bsregularizer1 = bsregularizer1
        self.bsregularizer2 = bsregularizer2

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.wtregularizer1 > 0:
            dl1 = np.ones_like(self.weights)
            dl1[self.weights < 0] = -1
            self.dweights += self.wtregularizer1 * dl1

        if self.wtregularizer2 > 0:
            self.dweights += 2 * self.wtregularizer2 * self.weights

        if self.bsregularizer1 > 0:
            dl1 = np.ones_like(self.biases)
            dl1[self.biases < 0] = -1
            self.dbiases += self.bsregularizer1 * dl1

        if self.bsregularizer2 > 0:
            self.dbiases += 2 * self.bsregularizer2 * self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)

class ReluActivation:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class SoftmaxActivation:
    def forward(self, inputs):
        expval = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        prob = expval / (np.sum(expval, axis=1, keepdims=True))
        self.output = prob

class CategoricalLoss:
    def forward(self, ypred, ytrue):
        samples = len(ypred)
        ypredclipped = np.clip(ypred, 1e-7, 1 - 1e-7)
        if len(ytrue.shape) == 1:
            prob = ypredclipped[range(samples), ytrue]
        elif len(ytrue.shape) == 2:
            prob = np.sum(ypredclipped * ytrue, axis=1)
        negloglikelihood = -np.log(prob)
        self.matrix = negloglikelihood
        self.mean = np.mean(negloglikelihood)

    def backward(self, dvalues, ytrue):
        samples = dvalues.shape[0]
        labels = dvalues.shape[1]
        if len(ytrue.shape) == 1:
            ytrue = np.eye(labels)[ytrue]
        self.dinputs = -ytrue / dvalues
        self.dinputs = self.dinputs / samples

class SoftmaxActivation_CategoricalLoss:
    def __init__(self):
        self.activation = SoftmaxActivation()
        self.loss = CategoricalLoss()

    def forward(self, inputs, ytrue):
        self.activation.forward(inputs)
        self.output = self.activation.output
        self.loss.forward(self.output, ytrue)
        return self.loss.mean

    def regularizationloss(self, layer):
        regularizationloss = 0
        if layer.wtregularizer1 > 0:
            regularizationloss += layer.wtregularizer1 * np.sum(np.abs(layer.weights))
        if layer.wtregularizer2 > 0:
            regularizationloss += layer.wtregularizer2 * np.sum(layer.weights * layer.weights)
        if layer.bsregularizer1 > 0:
            regularizationloss += layer.bsregularizer1 * np.sum(np.abs(layer.biases))
        if layer.bsregularizer2 > 0:
            regularizationloss += layer.bsregularizer2 * np.sum(layer.biases * layer.biases)
        return regularizationloss

    def backward(self, dvalues, ytrue):
        samples = dvalues.shape[0]
        if len(ytrue.shape) == 2:
            ytrue = np.argmax(ytrue, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), ytrue] -= 1
        self.dinputs = self.dinputs / samples

class gddecaymomentum:
    def __init__(self, L=1, decay=0., momentum=0.):
        self.L = L
        self.currentL = L
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def preupdateparams(self):
        if self.decay:
            self.currentL = self.L / (1. + self.decay * self.iterations)

    def updateparameter(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weightmomentums'):
                layer.weightmomentums = np.zeros_like(layer.weights)
                layer.biasmomentums = np.zeros_like(layer.biases)
            weightupdates = self.momentum * layer.weightmomentums - self.currentL * layer.dweights
            layer.weightmomentums = weightupdates
            biasupdates = self.momentum * layer.biasmomentums - self.currentL * layer.dbiases
            layer.biasmomentums = biasupdates
            layer.weights += weightupdates
            layer.biases += biasupdates
        else:
            layer.weights -= self.currentL * layer.dweights
            layer.biases -= self.currentL * layer.dbiases

    def postupdateparams(self):
        self.iterations += 1


def createbatches(X, y, batchsize):
    for i in range(0, X.shape[0], batchsize):
        yield X[i:i + batchsize], y[i:i + batchsize]

class OptimizerAdam:
    def __init__(self, L=0.001, decay=0., epsilon=1e-7, beta1=0.9, beta2=0.999):
        self.L = L
        self.currentL = L
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

    def preupdateparams(self):
        if self.decay:
            self.currentL = self.L * (1. / (1. + self.decay * self.iterations))

    def updateparams(self,layer):
        if not hasattr(layer, 'weightcache'):
            layer.weightmomentums = np.zeros_like(layer.weights)
            layer.weightcache = np.zeros_like(layer.weights)
            layer.biasmomentums = np.zeros_like(layer.biases)
            layer.biascache = np.zeros_like(layer.biases)

        layer.weightmomentums = self.beta1 * layer.weightmomentums + (1 - self.beta1) * layer.dweights
        layer.biasmomentums = self.beta1 * layer.biasmomentums + (1 - self.beta1) * layer.dbiases
        weightmomentumscorrected = layer.weightmomentums / (1 - self.beta1 ** (self.iterations + 1))
        biasmomentumscorrected = layer.biasmomentums / (1 - self.beta1 ** (self.iterations + 1))
        layer.weightcache = self.beta2 * layer.weightcache + (1 - self.beta2) * layer.dweights**2
        layer.biascache = self.beta2 * layer.biascache + (1 - self.beta2) * layer.dbiases**2
        weightcachecorrected = layer.weightcache / (1 - self.beta2 ** (self.iterations + 1))
        biascachecorrected = layer.biascache / (1 - self.beta2 ** (self.iterations + 1))
        layer.weights += -self.currentL * weightmomentumscorrected / (np.sqrt(weightcachecorrected) + self.epsilon)
        layer.biases += -self.currentL * biasmomentumscorrected / (np.sqrt(biascachecorrected) + self.epsilon)

    def postupdateparams(self):
        self.iterations+=1