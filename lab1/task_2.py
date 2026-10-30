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


def iterate(phi, x, eps):
    for k in range(MAX_ITER):
        x_new = phi(x)
        if(abs(x_new - x) < eps):
            return x_new, k + 1
        x = x_new
    return x, MAX_ITER


def max_dphi(dphi, a, b):
    x = np.linspace(a, b, 100)
    return np.max(np.abs(dphi(x)))


def plot_function(f, a, b, root, title):
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


# Main body

print("Уравнение 1: x^3 + 3x^2 + 6x - 1 = 0")

coefs1 = [1, 3, 6, -1]
low1 = lower_bound(coefs1)
up1 = upper_bound(coefs1)
print(f"Границы корней: [{low1}, {up1}]")

f1 = lambda x: x**3 + 3*x**2 + 6*x - 1
intervals1 = find_intervals(f1, low1, up1)
print(f"Отрезки с корнем: {intervals1}")

phi1 = lambda x: (1 - 3*x**2 - x**3) / 6
dphi1 = lambda x: -x - x**2 / 2

a1, b1 = intervals1[0]
print(f"max|phi'| на отрезке: {max_dphi(dphi1, a1, b1):.3f})")

root1, iters1 = iterate(phi1, (a1 + b1) / 2, EPS)
print(f"Корень: x = {root1:.4f} (итераций: {iters1}), f(x) = {f1(root1):.2e}\n")


print("Уравнение 2: x * 2^x = 1")

f2 = lambda x: x * 2**x - 1
intervals2 = find_intervals(f2, -1.0, 2.0)
print(f"Отрезки с корнем: {intervals2}")

phi2 = lambda x: 2 ** (-x)
dphi2 = lambda x: -np.log(2) * 2 ** (-x)

a2, b2 = intervals2[0]
print(f"max|phi'| на отрезке: {max_dphi(dphi2, a2, b2):.3f}")

root2, iters2 = iterate(phi2, (a2 + b2) / 2, EPS)
print(f"Корень: x = {root2:.4f} (итераций: {iters2}), f(x) = {f2(root2):.2e}")


# Графики
plot_function(f1, low1, up1, root1, "x^3 + 3x^2 + 6x - 1 = 0")
plot_function(f2, -1, 2, root2, "x * 2^x - 1 = 0")