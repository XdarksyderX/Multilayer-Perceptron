from .ActivationFunction import ActivationFunction
import numpy as np

class Sigmoid(ActivationFunction):
	@staticmethod
	def activate(x):
		return 1/(1 + np.exp(-x))

	@staticmethod
	def derivative(x):
		s = Sigmoid.activate(x)
		return s * (1 - s)
