from neural_network.core import Perceptron, Dense

layers = [
    Dense(2, use_bias=True),
    Dense(1, activation_function='sigmoid'),
]
perceptron = Perceptron(layers, learning_rate=0.2)

training_dataset = [[-0.9, -0.9], [-0.9, 1], [1, -0.9], [1, 1]]
expected_results = [[0], [0], [0], [1]]

perceptron.train(training_dataset, expected_results, batch_size=1, epochs=1000, shuffle=True, normalize=True)

print("LOGICAL CONJUNCTION TASK")
for expected_result, data in zip(expected_results, training_dataset):
    print(f"input={data} expected={expected_result} actual={perceptron.predict(data)}")


layers = [
    Dense(2),
    Dense(10, activation_function='relu'),
    Dense(1, activation_function='sigmoid'),
]
multilayer_perceptron = Perceptron(layers, learning_rate=0.2)

training_dataset = [[1, 1], [1, 2], [2, 1], [2, 2]]
expected_results = [[0], [1], [1], [0]]

multilayer_perceptron.train(training_dataset, expected_results, batch_size=1, epochs=1000, shuffle=True)

print("\nXOR TASK")
for expected_result, data in zip(expected_results, training_dataset):
    print(f"input={data} expected={expected_result} actual={perceptron.predict(data)}")

