import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

x=np.random.randn(1000,100) # Data(count:1000x100), 정규분포
node_num=100 # nodes per layer
hidden_layer_size=5
activations={}

for i in range(hidden_layer_size):
    # 0 ~ 4
    if i!=0:
        x=activations[i-1]
    w=np.random.randn(node_num, node_num)*1
    a=np.dot(x,w)
    z=sigmoid(a)
    activations[i]=z

for i, a in activations.items():
    # key, value
    plt.subplot(1,len(activations),i+1)
    plt.title(str(i+1)+"-layer")
    plt.hist(a.flatten(),30,range=(0,1))
plt.show()



