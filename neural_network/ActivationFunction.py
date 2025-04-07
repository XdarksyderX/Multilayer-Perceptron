from abc import ABC, abstractmethod

class ActivationFunction(ABC):

	@staticmethod
	@abstractmethod
	def activate(x):
		pass

	@staticmethod
	@abstractmethod
	def get_limit(input_size, size):
		pass
