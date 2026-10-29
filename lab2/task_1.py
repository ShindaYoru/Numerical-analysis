import copy
import time
import numpy as np
DET_EPS = 1e-12
MAX_ITER = 30

def determinant(matrix):
    if(len(matrix)==1):
        return matrix[0][0];
    if(len(matrix)==2):
        return matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0]
    total = 0
    for i in range(len(matrix)):
        total += (-1)**i*matrix[0][i]*determinant(cut_minor(matrix, i))
    return total
    
def cut_minor(matrix, element):
    size = len(matrix)-1
    temp_matrix = copy.deepcopy(matrix)
    temp_matrix.pop(0)
    for i in range(size):
        temp_matrix[i].pop(element)
    return temp_matrix


def cramer(A,b):
    d = determinant(A)
    if(abs(d) < DET_EPS):
        print("Система не имеет единственного решения")
        return None
    
    answer = []
    
    for i in range(len(A)):
        temp_matrix = copy.deepcopy(A)
        for j in range(len(A)):
            temp_matrix[j][i] = b[j]
        answer.append(determinant(temp_matrix)/d)

    return answer


def seidel(A, b, eps):
    x = []
    n = len(A)
    for i in range(n):
        x.append(b[i]/A[i][i])
    for k in range(MAX_ITER):
        x_old = copy.deepcopy(x);
        for i in range(n):
            s = sum(A[i][j]*x[j] for j in range(n) if j != i)
            x[i] = (b[i]-s)/A[i][i]
        diff = max(abs(x[i]-x_old[i]) for i in range(n))

        if diff < eps:
            return x, k+1
    return x, MAX_ITER


def is_diag_dominant(A):
    n = len(A)
    for i in range(n):
        diag = abs(A[i][i])
        res = sum(abs(A[i][j]) for j in range(n) if j!= i)
        if (diag < res):
            return False
    return True

# Main body
A = [[3,-0.5, 0.5, -1],
     [1,  5, -1,   -2],
     [2, -1,  4, -0.5],
     [1,  2, -1,    5]]

b = [12, 0.5, -6.6, 13]

start = time.time()
x = cramer(A, b)
end = time.time()
cramer_time = end - start

start = time.time()
x_s, iters = seidel(A, b, 1e-4)
end = time.time()
seidel_time = end - start

print("Диагонально доминантна?:", is_diag_dominant(A))

print("Крамера: ", x)
print("Зейдель: ", x_s)
print("Невязка:", np.max(abs(np.array(A) @ np.array(x) - b)))
print("Крамер в ", cramer_time/seidel_time, " раз медленее, чем Зейдель")
print("Итераций ", iters)
