from neural_network.core import Perceptron, Dense

layers = [
    Dense(2, use_bias=False),
    Dense(1, activation_function='sigmoid'),
]
perceptron = Perceptron(layers, learning_rate=0.1)

training_dataset = [[-0.9, -0.9], [-0.9, 1], [1, -0.9], [1, 1]]
expected_results = [[0], [0], [0], [1]]

perceptron.train(training_dataset, expected_results, batch_size=1, epochs=1000, shuffle=True, normalize=False)

for expected_result, train in zip(expected_results, training_dataset):
    print(expected_result, *perceptron.predict(train))

