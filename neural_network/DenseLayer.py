from random import random
import numpy as np

class DenseLayer:
	def __init__(self, size, activation):
		self.size = size
		self.activation = activation
		self.weights = None
		self.biases = None
		self.input_size = None

	def initialize(self, input_size):
		self.input_size = input_size
		limit = self.activation.get_limit(self.input_size, self.size)
		weights = np.random.uniform(-limit, limit, size=(input_size, self.size))
		biases = np.zeros((1, self.size))

