"""
Normalization
Sigmoid, ReLU
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def normalization(a):
    mean = np.mean(a)
    stdev = np.sqrt( np.sum((a-mean)**2) / a.size )
    b = (a-mean) / stdev
    return b

#Menu
print("1. sigmoid")
print("2. sigmoid + Xavier")
print("3. sigmoid + Normalization")
print("4. ReLU")
print("5. ReLU + Xavier")
print("6. ReLU + He")
print("7. ReLU + Normalization")

print("select menu ... ", end="")
sel=int(input())

x = np.random.randn(1000, 100)  # Data(count:1000x100), 정규분포
init_data = x

node_num = 100  # nodes per layer
hidden_layer_size = 5
activations = {}

for i in range(hidden_layer_size):
    # 0 ~ 4
    if i != 0:
        x = activations[i - 1]

    if sel==1 or sel==3 or sel==4 or sel==7:
        w = np.random.randn(node_num, node_num)
    elif sel==2 or sel==5:
        w = np.random.randn(node_num, node_num) * np.sqrt(1/node_num) # Xavier
    elif sel == 6:
        w = np.random.randn(node_num, node_num) * np.sqrt(2/node_num) # He

    #Process
    a = np.dot(x, w)

    # Normalization
    if sel==3 or sel==7:
        a = normalization(a)

    if sel<=3:
        z = sigmoid(a)
    else:
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