import numpy as np

from .LossFunction import LossFunction

class MeanSquaredError(LossFunction):
	@staticmethod
	def compute(y_pred, y_true):
		return np.mean((y_true - y_pred) ** 2)
	
	@staticmethod
	def derivative(y_pred, y_true):
		return 2 * (y_pred - y_true) / y_true.shape[0]
