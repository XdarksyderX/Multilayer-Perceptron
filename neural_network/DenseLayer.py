from random import random
import numpy as np

from .activation_functions import *
from .weights_initializer import *

ACTIVATION_FUNCTIONS = {
	'sigmoid': Sigmoid,
}

WEIGHTS_INITIALIZER = {
	'heUniform': HeUniform
}

class DenseLayer:
	def __init__(self, size, activation, weights_initializer):
		self.size = size
		self.activation = ACTIVATION_FUNCTIONS[activation]
		self.weights_initializer = WEIGHTS_INITIALIZER[weights_initializer]
		self.weights = None
		self.biases = None
		self.input_size = None

		#Backpropagation
		self.input = None
		self.Z = None
		self.dW = None
		self.db = None

	def initialize(self, input_size):
		self.input_size = input_size
		self.weights = self.weights_initializer.get_random_weights(self.input_size, self.size)
		self.biases = np.zeros((1, self.size))

	def forward(self, X):
		self.input = X
		Z = X @ self.weights + self.biases
		self.Z = Z
		A = self.activation.activate(Z)
		return A

	def backward(self, dA):
		dZ = dA * self.activation.derivative(self.Z)
		self.dW = self.input.T @ dZ
		self.db = np.sum(dZ, axis=0, keepdims=True)
		dA_prev = dZ @ self.dW.T
		return dA_prev

	def adjust(self, lr):
		self.weights -= lr * self.dW
		self.biases -= lr * self.db