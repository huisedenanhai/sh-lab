import math
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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


def mat_rot_z(rz):
    s, c = math.sin(rz), math.cos(rz)
    return np.array([
        [c, s, 0],
        [-s, c, 0],
        [0, 0, 1],
    ])


def mat_rot_x(rx):
    s, c = math.sin(rx), math.cos(rx)
    return np.array([
        [1, 0, 0],
        [0,  c, s],
        [0, -s, c],
    ])


def mat_rot_zxz(rz0, rx, rz1):
    return mat_rot_z(rz1) @ mat_rot_x(rx) @ mat_rot_z(rz0)


def rotate_l1(m, l1):
    # -f3 * x - f1 * y + f2 * z
    # dot([-f3, -f1, f2], [x, y, z])
    f1, f2, f3 = l1
    m_f3, m_f1, f2 = np.array([-f3, -f1, f2]) @ m
    return np.array([-m_f1, f2, -m_f3])


def rotate_l2(m, l2):
    # similar to http://filmicworlds.com/blog/simple-and-fast-spherical-harmonic-rotation/
    # for any direction d, we have
    # l2' * sh(d) = l2 * sh(m * d)
    # To solve l2', just pick some given directions d and solve the linear system
    # We choose the same basic directions as those presented in that blog, and let
    # A = sh(*ns)
    # B = sh(*(m * d))
    # Then
    # l2' = l2 * B * A^-1
    # A^-1 can be precomputed and sparse
    k = 1.0 / math.sqrt(2.0)
    ns = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [k, k, 0],
        [k, 0, k],
        [0, k, k],
    ]).T
    # A = sh_l2(*ns)
    k0 = 0.915291233
    k1 = 1.83058247
    k2 = 1.58533092
    inv_A = np.array([
        [0, -k0, 0, k0, k1],
        [k0, 0, k2, k0, k0],
        [k1, 0, 0, 0, 0],
        [0, 0, 0, -k1, 0],
        [0, -k1, 0, 0, 0]
    ])
    B = sh_l2(*(m @ ns))
    return l2 @ B @ inv_A


# Calculate the projected factors for l(m * x).
# To rotate the function it self with matrix m, one should calculate the projected factors for l(m.T * x)
def rotate_sh(m, l):
    return np.array([
        l[0],
        *rotate_l1(m, l[1:4]),
        *rotate_l2(m, l[4:9]),
    ])


def sample_basis(resolution):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    return x, y, z, eval_sh(x, y, z)


def set_up_axis(ax, r=0.5):
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


# need to hold reference to slider widgets
g_sliders = []


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


def plot_sh(factor, rotate=False, fixed_range=None):
    x, y, z, sh = sample_basis(30)
    fig = plt.figure()
    main_rect = [0.0, 0.2, 1.0, 0.8] if rotate else [0, 0, 1, 1]
    ax = fig.add_axes(main_rect, projection='3d')

    def update_factors(f):
        ax.clear()
        if fixed_range != None:
            set_up_axis(ax, fixed_range)

        l = np.tensordot(f, sh, axes=([0], [0]))
        v = np.abs(l)
        col = value_colors(l)
        ax.plot_surface(x * v, y * v, z * v, facecolors=col)

    update_factors(factor)

    if rotate:
        axes = [fig.add_axes([0.15, b, 0.75, 0.03]) for b in [0.15, 0.1, 0.05]]
        rot = [0, 0, 0]
        valmaxes = [2.0 * PI, PI, 2.0 * PI]
        labels = ['Rotate Z0', 'Rotate X', 'Rotate Z1']
        sliders = [
            Slider(
                ax=axes[i],
                label=labels[i],
                valmin=0,
                valmax=valmaxes[i],
                valinit=rot[i],
            ) for i in range(3)
        ]

        def update_func(i):
            def update(val):
                rot[i] = val
                m = mat_rot_zxz(*rot).T
                update_factors(rotate_sh(m, factor))
            return update

        for i in range(3):
            sliders[i].on_changed(update_func(i))

        global g_sliders
        g_sliders += sliders


def zonal_to_full(z):
    f = np.zeros(9)
    f[0], f[2], f[6] = z
    return f


def full_to_zonal(s):
    return np.array([s[0], s[2], s[6]])


def cos_lob_zonal():
    return np.array([
        SQRT_PI / 2.0,
        math.sqrt(PI / 3.0),
        math.sqrt(5.0 * PI / 64.0),
    ])


def const_lob_zonal(c):
    return np.array([c * 2.0 * SQRT_PI, 0.0, 0.0])


def cos_lob():
    return zonal_to_full(cos_lob_zonal())


def const_lob(c):
    return zonal_to_full(const_lob_zonal(c))


def conv_zonal(v, z):
    return np.array([
        v[l * (l + 1) + m] * z[l] * math.sqrt(4.0 * PI / (2.0 * l + 1.0))
        for l in range(0, 3)
        for m in range(-l, l + 1)
    ])


def sinc_attenuated(l, w):
    if l == 0:
        return 1.0
    elif l >= w:
        return 0.0

    x = PI * l / w
    f = math.sin(x) / x

    # The factor is actually differ from that provided in the paper 'Deringing spherical harmonics', Peter-Pike Sloan.
    # The attenuation here is slightly stronger.
    # It DOES make significant difference when deringing cos convolved delta function.
    # When the attenuation is applied, the result is much sharper, though light leaking accurs on the unlitted backside,
    # which matches the teaser image of the paper.
    # The paper does not tell precisely how the attenuation is calculated, I have to guess.
    # The paper says he uses sinc^4 but the factor in the table matches with sinc for unattenuated levels.
    # It might be a waste of time to figure out the exact implmentation used by the author :(
    # Filament also implements this. https://github.com/google/filament/blob/main/libs/ibl/src/CubemapSH.cpp
    # They use sinc^4 rather than the LUT provided by the paper.
    # For current attenuation factor, cutoff band 5 already makes cos convolved delta function non-negative.
    if l == 1:
        a = 1.0 - min(max(0.0, (11.0 - w) * 0.015), 0.1)
        f *= a
    return f


def window(v, w):
    scale = np.array([
        sinc_attenuated(l, w)
        for l in range(0, 3)
        for _ in range(-l, l + 1)
    ])
    return scale * v


def plot_2d_sh(ss, phi_min, phi_max):
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.15, 0.8, 0.8])

    def update(theta):
        ax.clear()
        u = np.linspace(phi_min, phi_max, 100)
        x = np.sin(u) * np.cos(theta)
        y = np.sin(u) * np.sin(theta)
        z = np.cos(u)
        b = eval_sh(x, y, z)
        ax.grid()
        for s in ss:
            v = np.tensordot(s, b, axes=([0], [0]))
            ax.plot(u, v)

    update(0)

    slider = Slider(
        ax=fig.add_axes([0.1, 0.05, 0.8, 0.03]),
        label='Theta',
        valmin=0,
        valmax=2.0 * PI,
        valinit=0.0,
    )
    slider.on_changed(update)
    g_sliders.append(slider)


def print_window_weight_table():
    for l in range(1, 3):
        ws = [str(sinc_attenuated(l, w))
              for w in [16.7, 11.3, 10.0, 9.0, 7.0, 5.6]]
        print(', '.join(ws))


def plot_delta():
    s = eval_sh(0.0, 0, 1)
    s_conv = conv_zonal(s, cos_lob_zonal())
    s_conv_windowed = window(s_conv, 5)
    plot_sh(s_conv_windowed)
    plot_2d_sh([s_conv, s_conv_windowed], 0.0, PI)


def safe_normalize(v, v0):
    n = np.linalg.norm(v)
    if n == 0:
        return v0
    return v / n


def rotate_primary_linear_dir_to_z(s):
    z = np.array([-s[3], -s[1], s[2]])
    z = safe_normalize(z, np.array([0, 0, 1]))
    x = np.cross(z, np.array([0, 0, 1]))
    x = safe_normalize(x, np.array([1, 0, 0]))
    y = safe_normalize(np.cross(z, x), np.array([0, 1, 0]))

    # rotate the primary linear direction to z
    sz = rotate_sh(np.array([x, y, z]).T, s)

    return sz


def estimate_lower_bound(s):
    s = rotate_primary_linear_dir_to_z(s)
    # zonal part
    f_5_4_s6 = math.sqrt(5.0) / (4.0 * SQRT_PI) * s[6]
    a = f_5_4_s6 * 3.0
    b = math.sqrt(3.0) / (2.0 * SQRT_PI) * s[2]
    c = s[0] / (2.0 * SQRT_PI) - f_5_4_s6

    # l = 2, m = 2
    q2 = math.sqrt(s[4] * s[4] + s[8] * s[8])
    d2 = math.sqrt(15) * q2 / (4.0 * SQRT_PI)
    a = a + d2
    c = c - d2

    z_min = min(a - b + c, a + b + c)
    if a > 0:
        z0 = -b / (2.0 * a)
        if z0 >= -1 and z0 <= 1:
            z_min = a * z0 * z0 + b * z0 + c

    # l = 2, m = 1
    q1 = math.sqrt(s[5] * s[5] + s[7] * s[7])
    d1 = math.sqrt(15) * q1 / (2.0 * SQRT_PI)
    z_min = z_min - d1 * 0.5

    # alreay positive, do early return
    if z_min >= 0:
        return z_min

    # following the paper 'Deringing spherical harmonics', we can solve for a tighter bound when z_min might goes negative
    def func(x):
        return a * x * x + b * x + c + d1 * x * math.sqrt(max(0.0, 1 - x * x))

    def func_d(x):
        z = math.sqrt(max(0.0, 1 - x * x))
        return 2.0 * a * x + b + d1 * z - d1 * x * x / z

    def func_dd(x):
        z = math.sqrt(max(0.0, 1 - x * x))
        return 2.0 * a - d1 * x / z - d1 * (2 * x * z * z + x * x * x) / (z * z * z)

    def inc(x):
        return -func_d(x) / func_dd(x)

    z = -1.0 / math.sqrt(2.0)
    dz = 10000
    while abs(z) <= 1 and abs(dz) > 1e-5:
        z_min = func(z)
        dz = inc(z)
        z = z + dz

    if abs(z) > 1:
        z_min = min(func(-1), func(1))

    return z_min


def find_window(s, w_min, w_max):
    for _ in range(16):
        if w_max - w_min <= 1e-4:
            break
        w = (w_min + w_max) / 2.0
        lower_bound = estimate_lower_bound(window(s, w))
        if lower_bound < 0:
            w_max = w
        else:
            w_min = w

    return w_min


def deringing(s):
    w = find_window(s, 1, 16)
    print(f'Window {w}')
    return window(s, w)


def test_lower_bound():
    s = np.random.rand(9)
    lower_bound = estimate_lower_bound(s)
    plot_2d_sh([s, const_lob(lower_bound)], 0.0, PI)


def test_rotate():
    s = np.random.rand(9)
    plot_sh(s, True, 0.5)
    plot_sh(rotate_primary_linear_dir_to_z(s), True, 0.5)


def test_deringing():
    s = np.random.rand(9)
    plot_sh(deringing(s))
    plot_sh(deringing(conv_zonal(eval_sh(0, 0, 1), cos_lob_zonal())))


if __name__ == '__main__':
    plot_basis()
    plot_delta()
    test_lower_bound()
    test_rotate()
    test_deringing()

    plt.show()
