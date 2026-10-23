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


# Отделение корней и метод итерации

def find_intervals(f, a, b, h=0.1):
    intervals = []
    steps = int(round((b - a) / h))
    for i in range(steps):
        x0 = a + i * h
        x1 = a + (i + 1) * h
        if f(x0) * f(x1) <= 0:
            intervals.append((x0, x1))
    return intervals

def iterate(phi, x, eps):
    for k in range(1, 1000):
        x_new = phi(x)               
        if abs(x_new - x) < eps:     
            return x_new, k
        x = x_new                    
    return None

# Уравнение 1: x^3 + 3x^2 + 6x - 1 = 0

print("Уравнение 1: x^3 + 3x^2 + 6x - 1 = 0")

coefs = [1, 3, 6, -1]

up = upper_bound(coefs)
low = lower_bound(coefs)
print(f"Границы корней: [{low}, {up}]")

f1 = lambda x: x**3 + 3*x**2 + 6*x - 1
intervals1 = find_intervals(f1, low, up, 0.1)
print(f"Отрезки с корнем: {intervals1}")

# Итерационная форма: x = (1 - 3x^2 - x^3) / 6
phi1 = lambda x: (1 - 3*x**2 - x**3) / 6

a1, b1 = intervals1[0]
root1, iters1 = iterate(phi1, (a1 + b1) / 2, EPS)
print(f"Корень: x = {root1:.4f} (итераций: {iters1})\n")

# Уравнение 2: x * 2^x = 1 

print("Уравнение 2: x * 2^x = 1")

f2 = lambda x: x * 2**x - 1
intervals2 = find_intervals(f2, -1.0, 2.0, 0.1)
print(f"Отрезки с корнем: {intervals2}")

# Итерационная форма: x = 2^(-x)

phi2 = lambda x: 2 ** (-x)

a2, b2 = intervals2[0]
root2, iters2 = iterate(phi2, (a2 + b2) / 2, EPS)
print(f"Корень: x = {root2:.4f} (итераций: {iters2})")

# Графики

def plot_graph(f, a, b, root, title):
    x = np.linspace(a, b, 400)
    plt.figure(figsize=(8, 5))
    plt.plot(x, f(x), label="f(x)", linewidth=2)
    plt.axhline(0, color="black")
    plt.axvline(root, color="red", linestyle="--", label=f"корень ≈ {root:.4f}")
    plt.plot(root, 0, "ro")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid()
    plt.legend()
    plt.show()

plot_graph(f1, low, up, root1, "x^3 + 3x^2 + 6x - 1 = 0")
plot_graph(f2, -1, 2, root2, "x * 2^x - 1 = 0")