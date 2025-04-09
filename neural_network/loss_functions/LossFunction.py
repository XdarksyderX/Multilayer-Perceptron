from abc import ABC, abstractmethod

class LossFunction(ABC):

	@staticmethod
	@abstractmethod
	def compute(y_pred, y_true):
		pass

	@staticmethod
	@abstractmethod
	def derivative(y_pred, y_true):
		pass
