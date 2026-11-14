import copy
import math
import matplotlib.pyplot as plt

EPS = 0.001
MAX_ITER = 1000
DET_EPS = 1e-12


# Из лабораторной 1: определитель и метод Крамера

def determinant(matrix):
    if(len(matrix) == 1):
        return matrix[0][0]
    if(len(matrix) == 2):
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    total = 0
    for i in range(len(matrix)):
        total += (-1)**i * matrix[0][i] * determinant(cut_minor(matrix, i))
    return total


def cut_minor(matrix, element):
    size = len(matrix) - 1
    temp_matrix = copy.deepcopy(matrix)
    temp_matrix.pop(0)
    for i in range(size):
        temp_matrix[i].pop(element)
    return temp_matrix


def cramer(A, b):
    d = determinant(A)
    if(abs(d) < DET_EPS):
        print("Система не имеет единственного решения")
        return None

    answer = []
    for i in range(len(A)):
        temp_matrix = copy.deepcopy(A)
        for j in range(len(A)):
            temp_matrix[j][i] = b[j]
        answer.append(determinant(temp_matrix) / d)
    return answer


# Из лабораторной 2: отделение корней по смене знака

def find_intervals(f, a, b, h=0.1):
    intervals = []
    steps = int(round((b - a) / h))
    for i in range(steps):
        x0 = a + i * h
        x1 = a + (i + 1) * h
        if(f(x0) * f(x1) <= 0):
            intervals.append((x0, x1))
    return intervals


# Матрица Якоби считается численно: J[i][j] = dF_i / dx_j (центральная разность)

def jacobian(F, x, y, h=1e-5):
    fx1, fx2 = F(x + h, y), F(x - h, y)
    fy1, fy2 = F(x, y + h), F(x, y - h)
    return [[(fx1[i] - fx2[i]) / (2 * h), (fy1[i] - fy2[i]) / (2 * h)] for i in range(2)]


def mat_vec(A, v):
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def transpose(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]


def linspace(a, b, n):
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def modified_newton(F, x, y, eps):
    J0 = jacobian(F, x, y)
    if(abs(determinant(J0)) < DET_EPS):
        print("Якобиан в начальной точке вырожден, метод неприменим")
        return None
    for k in range(MAX_ITER):
        dx, dy = cramer(J0, F(x, y))
        x -= dx
        y -= dy
        if(max(abs(dx), abs(dy)) < eps):
            return x, y, k + 1
    return x, y, MAX_ITER


# q = ||E - J0^(-1) J(x*)||: при q < 1 модифицированный метод Ньютона сходится
def convergence_q(J0, J_star):
    c1 = cramer(J0, [J_star[0][0], J_star[1][0]])
    c2 = cramer(J0, [J_star[0][1], J_star[1][1]])
    return max(abs(1 - c1[0]) + abs(c2[0]), abs(c1[1]) + abs(1 - c2[1]))


def gradient_method(F, x, y, eps):
    decreasing = True
    phi = sum(f**2 for f in F(x, y))
    for k in range(MAX_ITER):
        Fv = F(x, y)
        J = jacobian(F, x, y)
        # J^(-1) F - поправка Ньютона, используем как оценку погрешности (eps / 2 - запас)
        if(abs(determinant(J)) > DET_EPS):
            dx, dy = cramer(J, Fv)
            if(max(abs(dx), abs(dy)) < eps / 2):
                return x, y, k, decreasing
        g = mat_vec(transpose(J), Fv)
        Jg = mat_vec(J, g)
        if(Jg[0]**2 + Jg[1]**2 == 0):
            print("Градиент равен нулю, но решение не найдено")
            return x, y, k, decreasing
        a = (g[0]**2 + g[1]**2) / (Jg[0]**2 + Jg[1]**2)
        x -= a * g[0]
        y -= a * g[1]
        phi_new = sum(f**2 for f in F(x, y))
        if(phi_new > phi):
            decreasing = False
        phi = phi_new
    return x, y, MAX_ITER, decreasing


def plot_system(curves, points, title, xlim=None, ylim=None):
    plt.figure(figsize=(7, 6))
    for xs, ys, label, color in curves:
        plt.plot(xs, ys, color=color, label=label)
    for px, py in points:
        plt.plot(px, py, "ko")
    if(xlim):
        plt.xlim(xlim)
    if(ylim):
        plt.ylim(ylim)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.legend()
    plt.show()


# Main body
print("б) sin(x+1) - y = 1.8, 2x + cos(y) = 2")

F_b = lambda x, y: [math.sin(x + 1) - y - 1.8, 2 * x + math.cos(y) - 2]

# Выражаем x из второго уравнения и подставляем в первое: получаем уравнение с одной неизвестной y.
# Смена знака у g_b означает, что система имеет решение.
g_b = lambda y: F_b((2 - math.cos(y)) / 2, y)[0]
intervals_b = find_intervals(g_b, -3, 0)
print(f"Отрезки с решением по y: {intervals_b}")

a_b, b_b = intervals_b[0]
y0_b = (a_b + b_b) / 2
x0_b = (2 - math.cos(y0_b)) / 2
print(f"Начальное приближение: ({x0_b:.3f}; {y0_b:.3f})")

x_b, y_b, iters_b = modified_newton(F_b, x0_b, y0_b, EPS)
q_b = convergence_q(jacobian(F_b, x0_b, y0_b), jacobian(F_b, x_b, y_b))
print(f"Решение (модифицированный Ньютон): x = {x_b:.4f}, y = {y_b:.4f} (итераций: {iters_b})")
print(f"Невязка: {F_b(x_b, y_b)[0]:.2e}, {F_b(x_b, y_b)[1]:.2e}")
print(f"Условие сходимости q = {q_b:.3f} < 1: {q_b < 1}\n")


print("г) tg(xy + 0.1) = x^2, 3x^2 + 2y^2 = 1")

F_g = lambda x, y: [math.tan(x * y + 0.1) - x**2, 3 * x**2 + 2 * y**2 - 1]

# Второе уравнение - эллипс: x = cos(t) / sqrt(3), y = sin(t) / sqrt(2).
# Смотрим знак первого уравнения вдоль эллипса: смена знака означает решение системы.
ellipse = lambda t: (math.cos(t) / math.sqrt(3), math.sin(t) / math.sqrt(2))
f_t = lambda t: F_g(*ellipse(t))[0]
intervals_g = find_intervals(f_t, 0, 2 * math.pi)
print(f"Смен знака на эллипсе: {len(intervals_g)}")

solutions_g = []
for t0, t1 in intervals_g:
    x0, y0 = ellipse((t0 + t1) / 2)
    x_g, y_g, iters_g, decreasing = gradient_method(F_g, x0, y0, EPS)
    solutions_g.append((x_g, y_g))
    print(f"Решение (градиентный): x = {x_g:.4f}, y = {y_g:.4f} (итераций: {iters_g}), "
          f"невязка: {F_g(x_g, y_g)[0]:.2e}, {F_g(x_g, y_g)[1]:.2e}")
    print(f"Φ убывает на каждом шаге: {decreasing}")


# Графики
xs1 = linspace(0, 1.5, 200)
ys2 = linspace(-2, 0, 200)
plot_system([(xs1, [math.sin(x + 1) - 1.8 for x in xs1], "sin(x+1) - y = 1.8", "blue"),
             ([(2 - math.cos(y)) / 2 for y in ys2], ys2, "2x + cos(y) = 2", "red")],
            [(x_b, y_b)], "Модифицированный метод Ньютона")

xs = linspace(-0.8, 0.8, 400)
ys = [(math.atan(x**2) - 0.1) / x for x in xs]
ys = [y if abs(y) <= 1 else float("nan") for y in ys]
ts = linspace(0, 2 * math.pi, 200)
plot_system([(xs, ys, "tg(xy+0.1) = x^2", "blue"),
             ([ellipse(t)[0] for t in ts], [ellipse(t)[1] for t in ts], "3x^2 + 2y^2 = 1", "red")],
            solutions_g, "Градиентный метод", xlim=(-0.8, 0.8), ylim=(-0.8, 0.8))
