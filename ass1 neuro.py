import numpy as np

def random_uniform(low, high, seed):
    seed = (seed * 9301 + 49297) % 233280
    return low + (seed / 233280.0) * (high - low)

seed = 42
w1 = random_uniform(-0.5, 0.5, seed)
w2 = random_uniform(-0.5, 0.5, w1)
w3 = random_uniform(-0.5, 0.5, w2)
w4 = random_uniform(-0.5, 0.5, w3)
w5 = random_uniform(-0.5, 0.5, w4)
w6 = random_uniform(-0.5, 0.5, w5)

b1 = 0.5
b2 = 0.7
i1, i2 = 0.05, 0.1
target_o1, target_o2 = 0.1, 0.99

def exp(x, terms=10):
    result = 1
    factorial = 1
    power = 1
    for i in range(1, terms):
        factorial *= i
        power *= x
        result += power / factorial
    return result

def tanh(x):
    ex = exp(x)
    e_neg_x = exp(-x)
    return (ex - e_neg_x) / (ex + e_neg_x)

net_h1 = w1 * i1 + w2 * i2 + b1
net_h2 = w3 * i1 + w4 * i2 + b1
out_h1 = tanh(net_h1)
out_h2 = tanh(net_h2)

net_o1 = w5 * out_h1 + w6 * out_h2 + b2
net_o2 = w5 * out_h1 + w6 * out_h2 + b2
out_o1 = tanh(net_o1)
out_o2 = tanh(net_o2)

E_o1 = 0.5 * (target_o1 - out_o1) ** 2
E_o2 = 0.5 * (target_o2 - out_o2) ** 2
E_total = E_o1 + E_o2

print("Output h1:", round(out_h1, 6), ", Output h2:", round(out_h2, 6))
print("Output o1:", round(out_o1, 6), ", Output o2:", round(out_o2, 6))
print("Loss E_o1:", round(E_o1, 6), ", Loss E_o2:", round(E_o2, 6))
print("Total Loss:", round(E_total, 6))