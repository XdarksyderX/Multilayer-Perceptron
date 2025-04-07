from .WeightsInitializer import WeightsInitializer
import numpy as np

class HeUniform(WeightsInitializer):
	@staticmethod
	def get_random_weights(input_size, output_size):
		limit = np.sqrt(6/input_size)
		return np.random.uniform(-limit, limit, size=(input_size, output_size))
