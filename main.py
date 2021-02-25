"""
Demonstration of the possibilities of this implementation
"""
from neural_network.core import Perceptron, Dense

layers = [
    Dense(2, use_bias=True),
    Dense(1, activation_function='sigmoid'),
]
perceptron = Perceptron(layers, learning_rate=0.5)


training_dataset = [[-0.9, -0.9], [-0.9, 1], [1, -0.9], [1.0, 1]]
expected_results = [[0], [0], [0], [1]]

perceptron.train(training_dataset, expected_results, batch_size=2,
                 epochs=1000, shuffle=True, normalize=False)

print("'Logical conjunction' task:")

for expected, test in zip(expected_results, training_dataset):
    actual = perceptron.predict(test)
    print(f"test_input={test} expected={expected} actual={actual}")


layers = [
    Dense(2, use_bias=True),
    Dense(6, activation_function='relu', use_bias=True),
    Dense(1, activation_function='sigmoid'),
]
multilayer_perceptron = Perceptron(layers, learning_rate=0.2)

training_dataset = [[1, 1], [1, 2], [2, 1], [2, 2]]
expected_results = [[0], [1], [1], [0]]

multilayer_perceptron.train(training_dataset, expected_results, batch_size=1,
                            epochs=1000, shuffle=False, normalize=True)

print("\n'Exclusive or (XOR)' task:")

for expected, test in zip(expected_results, training_dataset):
    actual = multilayer_perceptron.predict(test)
    print(f"test_input={test} expected={expected} actual={actual}")
