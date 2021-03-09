"""
File with two simple examples of neural network training.

The first neural network is trained to solve the problem of logical conjunction.
A two-layer perceptron is enough to solve this problem.

The second neural network solves the problem of exclusion or.
This problem already requires 3 layers.
"""
from neural_network.core import Perceptron, Dense
from neural_network.utils import make_batches


layers = [
    Dense(2, use_bias=True),
    Dense(1, activation_function='sigmoid'),
]
perceptron = Perceptron(layers, learning_rate=0.5)

training_dataset = [[-0.9, -0.9], [-0.9, 1], [1, -0.9], [1, 1]]
# It is necessary to pass the expected results in the list of lists,
# because the last layer may contain more than one output neuron
expected_results = [[0], [0], [0], [1]]

perceptron.train(training_dataset, expected_results, batch_size=2,
                 epochs=1000, shuffle=True, normalize=False)

print("'Logical conjunction (AND)' task:")

for test, expected in zip(training_dataset, expected_results):
    actual = perceptron.predict(test)
    print(f"\tTest={test} Expected={expected} Actual={actual}")

layers = [
    Dense(2),
    Dense(10, activation_function='relu'),
    Dense(1, activation_function='sigmoid'),
]
multilayer_perceptron = Perceptron(layers, learning_rate=0.9)

training_dataset = [[1, 1], [1, 2], [2, 1], [2, 2]]
# Alternative to creating a list of lists manually
expected_results = make_batches([0, 1, 1, 0], batch_size=1)

multilayer_perceptron.train(training_dataset, expected_results, batch_size=1,
                            epochs=1000, scope=(-1, 1))

print("\n'Exclusive or (XOR)' task:")

for test, expected in zip(training_dataset, expected_results):
    actual = multilayer_perceptron.predict(test)
    print(f"\tTest={test} Expected={expected} Actual={actual}")
