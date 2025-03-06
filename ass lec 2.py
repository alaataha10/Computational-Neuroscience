import math

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Initialize weights and biases
weights = {
    "w1": 0.15, "w2": 0.20, "w3": 0.25, "w4": 0.30,
    "w5": 0.40, "w6": 0.45, "w7": 0.50, "w8": 0.55
}
biases = {"b1": 0.35, "b2": 0.60}

# Inputs and targets
inputs = [0.05, 0.10]
targets = [0.01, 0.99]

# Forward pass: Compute hidden layer outputs
h1_input = inputs[0] * weights["w1"] + inputs[1] * weights["w3"] + biases["b1"]
h2_input = inputs[0] * weights["w2"] + inputs[1] * weights["w4"] + biases["b1"]

h1, h2 = sigmoid(h1_input), sigmoid(h2_input)

# Compute output layer values
o1_input = h1 * weights["w5"] + h2 * weights["w7"] + biases["b2"]
o2_input = h1 * weights["w6"] + h2 * weights["w8"] + biases["b2"]

o1, o2 = sigmoid(o1_input), sigmoid(o2_input)

# Calculate losses
loss_o1 = 0.5 * (targets[0] - o1) ** 2
loss_o2 = 0.5 * (targets[1] - o2) ** 2
total_loss = loss_o1 + loss_o2

# Print forward pass results
print(f"Hidden Layer Outputs: h1 = {round(h1, 6)}, h2 = {round(h2, 6)}")
print(f"Output Values: o1 = {round(o1, 6)}, o2 = {round(o2, 6)}")
print(f"Losses: E_o1 = {round(loss_o1, 6)}, E_o2 = {round(loss_o2, 6)}")
print(f"Total Loss: {round(total_loss, 6)}")

# Backpropagation: Calculate deltas
delta_o1 = (o1 - targets[0]) * o1 * (1 - o1)
delta_o2 = (o2 - targets[1]) * o2 * (1 - o2)

delta_h1 = (delta_o1 * weights["w5"] + delta_o2 * weights["w6"]) * h1 * (1 - h1)
delta_h2 = (delta_o1 * weights["w7"] + delta_o2 * weights["w8"]) * h2 * (1 - h2)

# Compute gradients
gradients = {
    "w1": delta_h1 * inputs[0], "w2": delta_h2 * inputs[0],
    "w3": delta_h1 * inputs[1], "w4": delta_h2 * inputs[1],
    "w5": delta_o1 * h1, "w6": delta_o1 * h2,
    "w7": delta_o2 * h1, "w8": delta_o2 * h2,
    "b1": delta_h1 + delta_h2, "b2": delta_o1 + delta_o2
}

# Update weights and biases using gradient descent
learning_rate = 0.1
for key in weights:
    weights[key] -= learning_rate * gradients[key]
for key in biases:
    biases[key] -= learning_rate * gradients[key]

# Print updated weights and biases
print("\nUpdated Weights and Biases:")
for key, value in weights.items():
    print(f"{key} = {round(value, 6)}")
for key, value in biases.items():
    print(f"{key} = {round(value, 6)}")