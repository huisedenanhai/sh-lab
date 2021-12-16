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


def eval_sh(x, y, z):
    if hasattr(x, 'shape'):
        shape = (1, *x.shape)
    else:
        shape = 1
    return np.concatenate((
        sh_l0() * np.ones(shape),
        sh_l1(x, y, z),
        sh_l2(x, y, z),
    ))


def sample_basis(resolution):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    return x, y, z, eval_sh(x, y, z)


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

    return cm.Reds(p) * mask(p) + cm.Blues(np.sign(n)) * mask(n)


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
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(x * v, y * v, z * v, facecolors=col)


def zonal_to_full(z):
    f = np.zeros(9)
    f[0], f[2], f[6] = z
    return f


def cos_lob_zonal():
    return np.array([
        SQRT_PI / 2.0,
        math.sqrt(PI / 3.0),
        math.sqrt(5.0 * PI / 64.0),
    ])


def cos_lob():
    return zonal_to_full(cos_lob_zonal())


def conv_zonal(v, z):
    return np.array([
        v[l * (l + 1) + m] * z[l] * math.sqrt(4.0 * PI / (2.0 * l + 1.0))
        for l in range(0, 3)
        for m in range(-l, l + 1)
    ])


def window(v, w):
    f = PI / w
    scale = np.array([
        # (math.sin(f * l) / (f * l)) ** 4 if l > 0 else 1
        [0.8915, 0.8030, 0.5904][l]
        for l in range(0, 3)
        for _ in range(-l, l + 1)
    ])
    print(scale)
    return scale * v


def plot_zonal(s, phi_min, phi_max):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    u = np.linspace(phi_min, phi_max, 100)
    x = np.sin(u)
    z = np.cos(u)
    y = np.zeros(u.size)
    b = eval_sh(x, y, z)
    v = np.tensordot(s, b, axes=([0], [0]))
    ax.grid()
    ax.plot(u, v)


if __name__ == '__main__':
    plot_basis()
    s = eval_sh(0, 0, 1)
    print(f'SH delta {s}')
    plot_sh(s)
    plot_sh(conv_zonal(s, cos_lob_zonal()))
    plot_sh(window(s, 2.6))
    plot_zonal(s, 0.0, PI)
    plt.show()
