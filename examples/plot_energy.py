import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math

if __name__ == "__main__":

    a = 0.0000000001
    c = 7.255197456936871

    t_expect = [1.3, 2.8]
    E2_expect = []
    for tx in t_expect:
        E2_expect.append(a * math.exp(tx * c))

    t = np.load("t.npy")
    E2 = np.load("E2.npy")

    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes()

    ax.plot(t, E2)
    ax.plot(t_expect, E2_expect)

    # Adding labels

    ax.set_yscale("log")
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy")
    ax.set_title("Field Energy")

    plt.savefig("field_energy.pdf", bbox_inches="tight")
    plt.show()

    potential_energy = np.load("potential_energy.npy")
    kinetic_energy = np.load("kinetic_energy.npy")
    total_energy = np.load("total_energy.npy")

    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes()

    ax.plot(t, potential_energy, label="Potential")
    ax.plot(t, kinetic_energy, label="Kinetic")
    ax.plot(t, total_energy, label="Total")

    ax.set_xlabel("Time")
    ax.set_ylabel("Energy")
    ax.set_title("Particle Energy")

    ax.legend()

    plt.savefig("particle_energy.pdf", bbox_inches="tight")
    plt.show()

    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes()

    ax.plot(t, total_energy, label="Total")

    ax.set_xlabel("Time")
    ax.set_ylabel("Energy")
    ax.set_title("Particle Energy")

    ax.legend()

    plt.savefig("total_energy.pdf", bbox_inches="tight")
    plt.show()
