import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math

if __name__ == "__main__":

    
    a = 10000.0
    c = 7.255197456936871
    
    t_expect = [0.025, 0.077]
    E2_expect = []
    for tx in t_expect:
        E2_expect.append(a * math.exp(tx * c))

    t = np.load("t.npy")
    E2 = np.load("E2.npy")


    fig = plt.figure(figsize =(16, 9))
    ax = plt.axes()
    
    ax.plot(t, E2)
    ax.plot(t_expect, E2_expect)

    # Adding labels

    ax.set_yscale('log')
    ax.set_xlabel('X-axis', fontweight ='bold')
    ax.set_ylabel('Y-axis', fontweight ='bold')

    # show plot
    plt.show()







