import math
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt

SQRT_PI = math.sqrt(math.pi)
PI = math.pi


def sh_l0():
    return 1.0 / (2.0 * SQRT_PI)


def sh_l1(x, y, z):
    f = math.sqrt(3.0) / (2.0 * SQRT_PI)
    return np.array([-y, z, -x]) * f


def sh_l2(x, y, z):
    f_15_2 = math.sqrt(15.0) / (2.0 * SQRT_PI)
    f_5_4 = math.sqrt(5.0) / (4.0 * SQRT_PI)
    f_15_4 = math.sqrt(15.0) / (4.0 * SQRT_PI)

    return np.array([
        f_15_2 * x * y,
        -f_15_2 * y * z,
        f_5_4 * (3.0 * z * z - 1),
        -f_15_2 * x * z,
        f_15_4 * (x * x - y * y)
    ])


def plot_shs():
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    l0 = sh_l0() * np.ones((np.size(u), np.size(v)))
    l1 = sh_l1(x, y, z)
    l2 = sh_l2(x, y, z)

    fig = plt.figure()

    def visualize(l, index):
        v = np.abs(l)

        ax = fig.add_subplot(3, 5, index, projection='3d')

        r = 0.5
        ax.set(
            xlim=[-r, r],
            ylim=[-r, r],
            zlim=[-r, r],
        )
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.plot_surface(x * v, y * v, z * v)

    visualize(l0, 3)
    for i in range(3):
        visualize(l1[i], i + 7)
    for i in range(5):
        visualize(l2[i], i + 11)

    plt.show()


if __name__ == '__main__':
    plot_shs()
