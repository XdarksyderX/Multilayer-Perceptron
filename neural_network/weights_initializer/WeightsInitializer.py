import numpy as np
from abc import ABC, abstractmethod

class WeightsInitializer(ABC):
	@staticmethod
	@abstractmethod
	def get_random_weights(input_size, output_size):
		pass