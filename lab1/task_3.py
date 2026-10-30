import math
import numpy as np
import matplotlib.pyplot as plt

EPS = 0.001
MAX_ITER = 1000


def horner(coefs, c):
    b = [coefs[0]]
    for i in range(1, len(coefs)):
        b.append(b[i-1] * c + coefs[i])
    return b

def upper_bound(coefs):
    if(coefs[0] < 0):
        coefs = [-a for a in coefs]
    c = 1
    while(not all(x >= 0 for x in horner(coefs, c))):
        c += 1
    return c


def lower_bound(coefs):
    n = len(coefs) - 1
    coefs_neg = [a * (-1)**(n - i) for i, a in enumerate(coefs)]
    return -upper_bound(coefs_neg)


def find_intervals(f, a, b, h=0.1):
    intervals = []
    steps = int(round((b - a) / h))
    for i in range(steps):
        x0 = a + i * h
        x1 = a + (i + 1) * h
        if(f(x0) * f(x1) <= 0):
            intervals.append((x0, x1))
    return intervals


def chord_method(f, d2f, a, b, eps):
    for k in range(MAX_ITER):
        x = a - f(a) * (b - a) / (f(b) - f(a))
        if(f(a) * d2f(a) > 0):
            step = abs(x - b)
            b = x
        else:
            step = abs(x - a)
            a = x
        if(step < eps):
            return x, k + 1
    return x, MAX_ITER


def start_point(f, d2f, a, b):
    if(f(a) * d2f(a) > 0):
        return a
    return b


def tangent_method(f, df, x0, eps):
    x = x0
    for k in range(MAX_ITER):
        x_new = x - f(x) / df(x)
        if(abs(x_new - x) < eps):
            return x_new, k + 1
        x = x_new
    return x, MAX_ITER



def combined_method(f, df, d2f, a, b, eps):
    for k in range(MAX_ITER):
        x_chord = a - f(a) * (b - a) / (f(b) - f(a))
        if(f(a) * d2f(a) > 0):
            x_tan = a - f(a) / df(a)
            a, b = x_tan, x_chord
        else:
            x_tan = b - f(b) / df(b)
            a, b = x_chord, x_tan
        if(abs(b - a) < eps):
            return (a + b) / 2, k + 1
    return (a + b) / 2, MAX_ITER


# Main body
print("а) x + lg(x) = 0.5")

f_a = lambda x: x + np.log10(x) - 0.5
d2f_a = lambda x: -1 / (x**2 * math.log(10))


intervals_a = find_intervals(f_a, 0.1, 10)
print(f"Отрезки с корнем: {intervals_a}")

a_a, b_a = intervals_a[0]
root_a, iters_a = chord_method(f_a, d2f_a, a_a, b_a, EPS)
print(f"Корень (метод хорд): x = {root_a:.4f} (итераций: {iters_a}), f(x) = {f_a(root_a):.2e}\n")


# б) x^3 + x - 3 = 0, метод касательных
print("б) x^3 + x - 3 = 0")

coefs_b = [1, 0, 1, -3]
low_b = lower_bound(coefs_b)
up_b = upper_bound(coefs_b)
print(f"Границы корней: [{low_b}, {up_b}]")

f_b = lambda x: x**3 + x - 3
df_b = lambda x: 3 * x**2 + 1
d2f_b = lambda x: 6 * x

intervals_b = find_intervals(f_b, low_b, up_b)
print(f"Отрезки с корнем: {intervals_b}")

a_b, b_b = intervals_b[0]
x0_b = start_point(f_b, d2f_b, a_b, b_b)
print(f"Начальное приближение: {x0_b:.1f}")

root_b, iters_b = tangent_method(f_b, df_b, x0_b, EPS)
print(f"Корень (метод касательных): x = {root_b:.4f} (итераций: {iters_b}), f(x) = {f_b(root_b):.2e}\n")



print("в) x^3 - 4x^2 + 2 = 0")

coefs_c = [1, -4, 0, 2]
low_c = lower_bound(coefs_c)
up_c = upper_bound(coefs_c)
print(f"Границы корней: [{low_c}, {up_c}]")

f_c = lambda x: x**3 - 4 * x**2 + 2
df_c = lambda x: 3 * x**2 - 8 * x
d2f_c = lambda x: 6 * x - 8

intervals_c = find_intervals(f_c, low_c, up_c)
print(f"Отрезки с корнями: {intervals_c}")

a_c, b_c = intervals_c[0]
root_c, iters_c = combined_method(f_c, df_c, d2f_c, a_c, b_c, EPS)
print(f"Корень (комбинированный метод): x = {root_c:.4f} (итераций: {iters_c}), f(x) = {f_c(root_c):.2e}")
