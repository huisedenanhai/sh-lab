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


def sample_basis(resolution):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    l0 = sh_l0() * np.ones((np.size(u), np.size(v)))
    l1 = sh_l1(x, y, z)
    l2 = sh_l2(x, y, z)
    return x, y, z, np.concatenate((np.array((l0, )), l1, l2), axis=0)


def set_up_axis(ax):
    r = 0.5
    ax.set(
        xlim=[-r, r],
        ylim=[-r, r],
        zlim=[-r, r],
    )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def value_colors(l):
    p = np.maximum(l, 0)
    n = np.maximum(-l, 0)

    def mask(v):
        return np.repeat(np.sign(v).reshape((*v.shape, 1)), 4, axis=2)

    return cm.Reds(p) * mask(p) + cm.Blues(n) * mask(n)


def plot_basis():
    x, y, z, sh = sample_basis(30)

    fig = plt.figure()

    def visualize(l, index):
        v = np.abs(l)

        ax = fig.add_subplot(3, 5, index, projection='3d')
        set_up_axis(ax)
        col = value_colors(l)

        ax.plot_surface(x * v, y * v, z * v, facecolors=col)

    visualize(sh[0], 3)
    for i in range(3):
        visualize(sh[i + 1], i + 7)
    for i in range(5):
        visualize(sh[i + 4], i + 11)


def plot_sh(factor):
    x, y, z, sh = sample_basis(100)
    l = np.tensordot(factor, sh, axes=([0], [0]))
    v = np.abs(l)
    col = value_colors(l)
    ax = plt.axes(projection='3d')
    ax.plot_surface(x * v, y * v, z * v, facecolors=col)


if __name__ == '__main__':
    # plot_basis()
    plot_sh(np.random.rand(9))
    plt.show()
