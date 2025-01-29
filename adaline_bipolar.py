# ADAptive LInear NEuron bipolar learning
def activator(value):
    return 1 if value >= 0 else -1


def gate(value):
    if value == "OR":
        return [[[-1, -1], [-1, 1], [1, -1], [1, 1]], [-1, 1, 1, 1]]
    elif value == "AND":
        return [[[-1, -1], [-1, 1], [1, -1], [1, 1]], [-1, -1, -1, 1]]
    # can add more gates as well here


def adaline_learning_train(value):
    weights = [0, 0]
    bias = 0
    learning_rate = 0.1

    X, y = gate(value)

    for epoch in range(1000):
        for i in range(len(X)):
            net_result = 0
            for j in range(len(weights)):
                net_result += weights[j] * X[i][j]
            net_result += bias
            error = y[i] - net_result
            bias += error * learning_rate
            for k in range(len(weights)):
                weights[k] += error * learning_rate * X[i][k]

    return bias, weights


def adaline_learning_test(value):
    bias, weight_list = adaline_learning_train(value)
    X, y = gate(value)
    predictions = []
    for i in range(len(X)):
        net_result = 0
        for j in range(len(weight_list)):
            net_result += weight_list[j] * X[i][j]
        net_result += bias
        predictions.append(net_result)

    return predictions


predicted_list = list(map(activator, adaline_learning_test("OR")))
print(predicted_list)

