# for the ADALINE learning

# initialize random weights and bias then iterate through the epochs to get the result of bias and weight then find the predictions

def activator(value):
    return 1 if value > 0.5 else 0  # 0 for bipolar and 0.5 for unipolar


bias = 0.1
weights = [0.1, 0.1]
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
output = [0, 1, 1, 1]

epochs = 100
learning_rate = 0.1
weighted_sum = 0

for epoch in range(epochs):
    for i in range(len(inputs)):
        weighted_sum = 0
        for j in range(len(weights)):
            weighted_sum += inputs[i][j] * weights[j]
        weighted_sum += bias

        error = output[i] - weighted_sum
        bias += learning_rate * error

        for k in range(len(weights)):
            weights[k] += error*learning_rate*inputs[i][k]


print(weights, bias)

predictions = []
value = 0
for i in range(len(inputs)):
    value = 0
    for j in range(len(weights)):
        value += inputs[i][j] * weights[j]
    value += bias
    predictions.append(value)

print(list(map(activator, predictions)))

