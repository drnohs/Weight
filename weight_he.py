"""
He by Kaiming He
ReLU
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0,x)

x = np.random.randn(1000, 100)  # Data(count:1000x100), 정규분포
init_data = x

node_num = 100  # nodes per layer
hidden_layer_size = 5
activations = {}

for i in range(hidden_layer_size):
    # 0 ~ 4
    if i != 0:
        x = activations[i - 1]

    #w = np.random.randn(node_num, node_num) * 0.01
    #w = np.random.randn(node_num, node_num) * np.sqrt(1/node_num) # Xavier, 0.1, Not Proper
    w = np.random.randn(node_num, node_num) * np.sqrt(2/node_num) # He, Proper, Wider

    a = np.dot(x, w)
    z = relu(a)
    activations[i] = z

# input Data
plt.subplot(1, len(activations) + 1, 1)
plt.title("Input")
plt.hist(init_data.flatten(), 50, range=(-1, 1))

for i, a in activations.items():
    # key, value
    plt.subplot(1, len(activations) + 1, i + 2)
    plt.title(str(i + 1) + "-layer")
    plt.hist(a.flatten(), 50, range=(0, 1))
plt.show()