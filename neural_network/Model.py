class Model:
	def __init__(self):
		self.layers = []
		self.built = False
	
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
