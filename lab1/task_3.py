import math
import numpy as np
import matplotlib.pyplot as plt

EPS = 0.001

# Схема Горнера и границы корней

def horner(coefs, c):
    len_coefs = len(coefs)
    b = [0.0] * len_coefs
    b[0] = coefs[0]
    for i in range(1, len_coefs):
        b[i] = b[i-1] * c + coefs[i]
    return b

def upper_bound(coefs):
    if coefs[0] < 0:
        coefs = [-a for a in coefs]
    c = 1
    while True:
        row = horner(coefs, c)
        if all(x >= 0 for x in row):
            return c
        c += 1

def lower_bound(coefs):
    n = len(coefs) - 1
    coefs = coefs[:]
    for i, a in enumerate(coefs):
        if (n - i) % 2 == 1:
            coefs[i] *= -1
    return -upper_bound(coefs)

def find_intervals(f, a, b, h=0.1):
    intervals = []
    steps = int(round((b - a) / h))
    for i in range(steps):
        x0 = a + i * h
        x1 = a + (i + 1) * h
        if f(x0) * f(x1) <= 0:
            intervals.append((x0, x1))
    return intervals


# Метод хорд

def chord_method(f, d2f, a, b, eps):
    history = []
    for k in range(1, 1000):
        x = a - f(a) * (b - a) / (f(b) - f(a))
        history.append((a, b, x))
        # неподвижен тот конец, где f(x) * f''(x) > 0
        if f(a) * d2f(a) > 0:
            step = abs(x - b)
            b = x
        else:
            step = abs(x - a)
            a = x
        if step < eps:
            return x, k, history
    return None


# Метод касательных

def start_point(f, d2f, a, b):
    if f(a) * d2f(a) > 0:
        return a
    return b

def tangent_method(f, df, x0, eps):
    history = []
    x = x0
    for k in range(1, 1000):
        x_new = x - f(x) / df(x)
        history.append((x, x_new))
        if abs(x_new - x) < eps:
            return x_new, k, history
        x = x_new
    return None


# Комбинированный метод

def combined_method(f, df, d2f, a, b, eps):
    history = []
    for k in range(1, 1000):
        x_chord = a - f(a) * (b - a) / (f(b) - f(a))
        if f(a) * d2f(a) > 0:
            x_tan = a - f(a) / df(a)
            history.append((a, b, x_chord, x_tan, True))
            a, b = x_tan, x_chord
        else:
            x_tan = b - f(b) / df(b)
            history.append((a, b, x_chord, x_tan, False))
            a, b = x_chord, x_tan
        if abs(b - a) < eps:
            return (a + b) / 2, k, history
    return None


# а) x + lg(x) = 0.5, метод хорд

print("а) x + lg(x) = 0.5")

f_a = lambda x: x + np.log10(x) - 0.5
d2f_a = lambda x: -1 / (x**2 * math.log(10))

# ОДЗ: x > 0, f'(x) = 1 + 1/(x ln10) > 0 -> корень один
intervals_a = find_intervals(f_a, 0.1, 10, 0.1)
print(f"Отрезки с корнем: {intervals_a}")

a_a, b_a = intervals_a[0]
root_chord, iters_chord, history_chord = chord_method(f_a, d2f_a, a_a, b_a, EPS)
print(f"Корень (метод хорд): x = {root_chord:.4f} (итераций: {iters_chord})\n")


# б) x^3 + x - 3 = 0, метод касательных

print("б) x^3 + x - 3 = 0")

coefs_b = [1, 0, 1, -3]
up_b = upper_bound(coefs_b)
low_b = lower_bound(coefs_b)
print(f"Границы корней: [{low_b}, {up_b}]")

f_b = lambda x: x**3 + x - 3
df_b = lambda x: 3 * x**2 + 1
d2f_b = lambda x: 6 * x

intervals_b = find_intervals(f_b, low_b, up_b, 0.1)
print(f"Отрезки с корнем: {intervals_b}")

a_b, b_b = intervals_b[0]
x0_b = start_point(f_b, d2f_b, a_b, b_b)
print(f"Начальное приближение: {x0_b:.1f}")

root_tangent, iters_tangent, history_tangent = tangent_method(f_b, df_b, x0_b, EPS)
print(f"Корень (метод касательных): x = {root_tangent:.4f} (итераций: {iters_tangent})\n")


# в) x^3 - 4x^2 + 2 = 0, комбинированный метод

print("в) x^3 - 4x^2 + 2 = 0")

coefs_c = [1, -4, 0, 2]
up_c = upper_bound(coefs_c)
low_c = lower_bound(coefs_c)
print(f"Границы корней: [{low_c}, {up_c}]")

f_c = lambda x: x**3 - 4 * x**2 + 2
df_c = lambda x: 3 * x**2 - 8 * x
d2f_c = lambda x: 6 * x - 8

intervals_c = find_intervals(f_c, low_c, up_c, 0.1)
print(f"Отрезки с корнями: {intervals_c}")

a_c, b_c = intervals_c[0]
root_combined, iters_combined, history_combined = combined_method(f_c, df_c, d2f_c, a_c, b_c, EPS)
print(f"Корень (комбинированный метод): x = {root_combined:.4f} (итераций: {iters_combined})")


# Графики

def draw_base(f, a, b):
    pad = 0.5 * (b - a)
    xs = np.linspace(a - pad, b + pad, 300)
    ys = [f(x) for x in xs]
    plt.figure(figsize=(9, 6))
    plt.plot(xs, ys, "b-", label="f(x)", linewidth=2)
    plt.axhline(0, color="black", linewidth=0.8)

def finish_plot(root, title):
    plt.axvline(root, color="red", linestyle="--", label=f"корень ≈ {root:.5f}")
    plt.plot(root, 0, "ro", markersize=10)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def plot_chord(f, a, b, root, history, title):
    draw_base(f, a, b)
    n = len(history)
    for i, (a_i, b_i, x_new) in enumerate(history):
        alpha = 0.3 + 0.7 * (i + 1) / n
        plt.plot([a_i, b_i], [f(a_i), f(b_i)], "--", color="orange", alpha=alpha,
                 linewidth=1.8, label="Хорды" if i == 0 else None)
        plt.plot(x_new, 0, "o", color="orange", alpha=alpha, markersize=6)
    finish_plot(root, title)

def plot_tangent(f, a, b, root, history, title):
    draw_base(f, a, b)
    n = len(history)
    for i, (x_curr, x_next) in enumerate(history):
        alpha = 0.3 + 0.7 * (i + 1) / n
        plt.plot([x_curr, x_next], [f(x_curr), 0], "-", color="green", alpha=alpha,
                 linewidth=1.8, label="Касательные" if i == 0 else None)
        plt.plot(x_next, 0, "o", color="green", alpha=alpha, markersize=6)
    finish_plot(root, title)

def plot_combined(f, a, b, root, history, title):
    draw_base(f, a, b)
    n = len(history)
    for i, (a_i, b_i, x_chord, x_tan, tan_from_a) in enumerate(history):
        alpha = 0.3 + 0.7 * (i + 1) / n
        plt.plot([a_i, b_i], [f(a_i), f(b_i)], "--", color="orange", alpha=alpha,
                 linewidth=1.8, label="Хорды" if i == 0 else None)
        x_start = a_i if tan_from_a else b_i
        plt.plot([x_start, x_tan], [f(x_start), 0], "-", color="green", alpha=alpha,
                 linewidth=1.8, label="Касательные" if i == 0 else None)
        plt.plot(x_chord, 0, "o", color="orange", alpha=alpha, markersize=6)
        plt.plot(x_tan, 0, "o", color="green", alpha=alpha, markersize=6)
    finish_plot(root, title)

plot_chord(f_a, a_a, b_a, root_chord, history_chord, "Метод хорд: x + lg(x) = 0.5")
plot_tangent(f_b, a_b, b_b, root_tangent, history_tangent, "Метод касательных: x^3 + x - 3 = 0")
plot_combined(f_c, a_c, b_c, root_combined, history_combined, "Комбинированный метод: x^3 - 4x^2 + 2 = 0")
