import numpy as np


# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Input data for XOR
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[1], [0], [0], [1]])

# Initialize parameters
input_neurons = 2   # Number of input neurons
hidden_neurons = 2  # Number of hidden neurons
output_neurons = 1  # Number of output neurons
learning_rate = 0.5  # Learning rate

# Initialize weights and biases
weights_input_hidden = np.random.uniform(-1, 1, (input_neurons, hidden_neurons))
weights_hidden_output = np.random.uniform(-1, 1, (hidden_neurons, output_neurons))
bias_hidden = np.random.uniform(-1, 1, (1, hidden_neurons))
bias_output = np.random.uniform(-1, 1, (1, output_neurons))

# Training
epochs = 10000
for epoch in range(epochs):
    # Forward pass
    hidden_input = np.dot(inputs, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)

    # Calculate error
    error = outputs - final_output

    # Backpropagation
    delta_output = error * sigmoid_derivative(final_output)
    delta_hidden = delta_output.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_output)

    # Update weights and biases
    weights_hidden_output += hidden_output.T.dot(delta_output) * learning_rate
    weights_input_hidden += inputs.T.dot(delta_hidden) * learning_rate
    bias_output += np.sum(delta_output, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(delta_hidden, axis=0, keepdims=True) * learning_rate


# Testing the model
print("Trained XNOR Gate Results:")
for i, input_data in enumerate(inputs):
    hidden_input = np.dot(input_data, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)

    print(f"Input: {input_data} Output: {np.round(final_output)}")
