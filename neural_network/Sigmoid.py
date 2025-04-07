from ActivationFunction import ActivationFunction
from math import exp, sqrt

class Sigmoid(ActivationFunction):

	@staticmethod
	def activate(x):
		return 1/(1 + exp(-x))

	@staticmethod
	def get_limit(input_size, size):
		return sqrt(6 / (input_size + size))
