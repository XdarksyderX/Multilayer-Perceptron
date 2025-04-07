from abc import ABC, abstractmethod

class ActivationFunction(ABC):

	@staticmethod
	@abstractmethod
	def activate(x):
		pass

	@staticmethod
	@abstractmethod
	def derivative(x):
		pass