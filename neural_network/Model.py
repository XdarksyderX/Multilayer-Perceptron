
from .loss_functions import MeanSquaredError
from concurrent.futures import ThreadPoolExecutor

LOSS_FUNCTIONS = {
	'mse': MeanSquaredError
}

class Model:
	def __init__(self, loss_function='mse'):
		self.layers = []
		self.built = False
		self.loss_function = LOSS_FUNCTIONS[loss_function]
	
	def add(self, layer):
		self.layers.append(layer)

	def createNetwork(self, layers):
		self.layers = layers

	def build(self, input_size):
		for layer in self.layers:
			layer.initialize(input_size)
			input_size = layer.size
		self.built = True

	def predict(self, X):
		if not self.built:
			self.build(X.shape[1])
		out = X
		for layer in self.layers:
			out = layer.forward(out)
		return out

	def train_step(self, X, y_true, lr):
		y_pred = self.predict(X)
		dA = self.loss_function.derivative(y_pred, y_true)
		
		for layer in reversed(self.layers):
			dA = layer.backward(dA)
		
		with ThreadPoolExecutor() as executor:
			executor.map(lambda l: l.adjust(lr), self.layers)
