from neural_network import Model, DenseLayer
import numpy as np

model = Model()

model.createNetwork([
	DenseLayer(10, activation='sigmoid', weights_initializer='heUniform'),
	DenseLayer(10, activation='sigmoid', weights_initializer='heUniform'),
	DenseLayer(10, activation='sigmoid', weights_initializer='heUniform')
])

X = np.random.rand(5, 3)

Y = model.predict(X)

print("INPUT")
print(X)

print("OUTPUT")
print(Y)

