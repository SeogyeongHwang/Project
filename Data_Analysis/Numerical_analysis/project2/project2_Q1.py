# 1번

import numpy as np
import matplotlib.pyplot as plt
import math

class find_matrix_solution:
  def __init__(self, lamb):
    self.A = np.array([[lamb, 3, 2, 1], [4, lamb, 7, 5], [8, 2, lamb, 2], [0, 1, 2, lamb]])
    self.b = np.array([[-3, 2, -2, 5]]).T
    return

  def inversion(self):
    A_inv = np.linalg.inv(self.A)
    x = A_inv @ self.b

    return x

  def NaiveGaussianElimination(self):
    n = np.size(self.b)
    Aug = np.concatenate((self.A, self.b), axis = 1)

    # Forward
    for k in range(0, n-1):
      for i in range(k+1, n):
       coeff = Aug[i, k] / Aug[k, k]
       Aug[i, k:n+1] = Aug[i, k:n+1] - coeff * Aug[k, k:n+1]

    # Backward
    x = np.zeros((n, 1))
    x[n-1] = Aug[n-1, n] / Aug[n-1, n-1]
    for i in range(n-2, -1, -1):
     x[i] = (Aug[i, n] - Aug[i, i+1:n] @ x[i+1:n]) / Aug[i, i]

    # true_relative error
    true_value = self.inversion()
    error = np.abs((x - true_value) / true_value)

    return x, error

  def GaussianElimination_Pivotting(self):
    n = np.size(self.b)
    Aug = np.concatenate((self.A, self.b), axis = 1)

    # pivotting
    max_idx = np.argmax(self.A[:,0])
    Aug[[0, max_idx], :] = Aug[[max_idx,0], :]

    # Forward
    for k in range(0, n-1):
      for i in range(k+1, n):
       coeff = Aug[i, k] / Aug[k, k]
       Aug[i, k:n+1] = Aug[i, k:n+1] - coeff * Aug[k, k:n+1]

    # Backward
    x = np.zeros((n, 1))
    x[n-1] = Aug[n-1, n] / Aug[n-1, n-1]
    for i in range(n-2, -1, -1):
     x[i] = (Aug[i, n] - Aug[i, i+1:n] @ x[i+1:n]) / Aug[i, i]

    # true_relative error
    true_value = self.inversion()
    error = np.abs((x - true_value) / true_value)

    return x, error

  def Gauss_Seidal(self, max_iter = 100, eps = 0.01):
    # check convergence for Gauss-Seidel method
    for i in range(len(self.b)):
      sum_row = 0
      for j in range(len(self.A)):
        if (j != i):
           sum_row += np.abs(self.A[i][j])
      if np.abs(self.A[i][i]) < sum_row:
        return ("This is not diagonally dominant")

    n = np.size(self.b)
    x_old = np.zeros((n, 1))
    x_new = np.zeros((n, 1))

    for iter in range(max_iter):
      for i in range(n):
        x_new[i] = (self.b[i] - ((self.A[i, :i] @ x_new[:i]) + (self.A[i, i+1:] @ x_old[i+1:]))) / self.A[i, i]

      x_old = x_new.copy()

      if np.sum((self.A @ x_new - self.b.T)**2)**0.5 < eps or iter == max_iter-1:
        return x_new

    return x_new

  def GaussianElimination_roundoff(self, decimals=10):
    n = np.size(self.b)
    Aug = np.concatenate((self.A, self.b), axis = 1)

    # Forward
    for k in range(0, n-1):
      for i in range(k+1, n):
       coeff = Aug[i, k] / Aug[k, k]
       Aug[i, k:n+1] = Aug[i, k:n+1] - coeff * Aug[k, k:n+1]
       Aug = np.round(Aug, decimals=decimals)

    # Backward
    x = np.zeros((n, 1))
    x[n-1] = Aug[n-1, n] / Aug[n-1, n-1]
    x[n-1] = np.round(x[n-1], decimals = decimals)
    for i in range(n-2, -1, -1):
     x[i] = (Aug[i, n] - Aug[i, i+1:n] @ x[i+1:n]) / Aug[i, i]
     x[i] = np.round(x[i], decimals = decimals)

    # true_relative error
    true_value = self.inversion()
    error = np.abs((x - true_value) / true_value)

    return x, error

  def GaussianElimination_Pivotting_roundoff(self, decimals=10):
    n = np.size(self.b)
    Aug = np.concatenate((self.A, self.b), axis = 1)

    # pivotting
    max_idx = np.argmax(self.A[:,0])
    Aug[[0, max_idx], :] = Aug[[max_idx,0], :]

    # Forward
    for k in range(0, n-1):
      for i in range(k+1, n):
        coeff = Aug[i, k] / Aug[k, k]
        Aug[i, k:n+1] = Aug[i, k:n+1] - coeff * Aug[k, k:n+1]
        Aug = np.round(Aug, decimals = decimals)

    # Backward
    x = np.zeros((n, 1))
    x[n-1] = Aug[n-1, n] / Aug[n-1, n-1]
    x[n-1] = np.round(x[n-1], decimals=decimals)
    for i in range(n-2, -1, -1):
      x[i] = (Aug[i, n] - Aug[i, i+1:n] @ x[i+1:n]) / Aug[i, i]
      x[i] = np.round(x[i], decimals=decimals)

    # true_relative error
    true_value = self.inversion()
    error = np.abs((x - true_value) / true_value)

    return x, error

  def Gauss_Seidal_roundoff(self, max_iter = 100, eps = 0.01, decimals=10):
    # check convergence for Gauss-Seidel method
    for i in range(len(self.b)):
      sum_row = 0
      for j in range(len(self.A)):
        if (j != i):
           sum_row += np.abs(self.A[i][j])
      if np.abs(self.A[i][i]) < sum_row:
        return ("This is not diagonally dominant")

    n = np.size(self.b)
    x_old = np.zeros((n, 1))
    x_new = np.zeros((n, 1))

    for iter in range(max_iter):
      for i in range(n):
        x_new[i] = (self.b[i] - ((self.A[i, :i] @ x_new[:i]) + (self.A[i, i+1:] @ x_old[i+1:]))) / self.A[i, i]
        x_new[i] = np.round(x_new, decimals = decimals)

      x_old = x_new.copy()

      if np.sum((self.A @ x_new - self.b.T)**2)**0.5 < eps or iter == max_iter-1:
        return x_new

    return x_new

if __name__ == "__main__":
    n = 1
    lamb = 10**(-n)
    q1 = find_matrix_solution(lamb)

    # 1
    print("1. inversion ")
    print(q1.inversion())

    # 2-A
    print("\n2-A. Naive Gaussian elimination")
    sol, error = q1.NaiveGaussianElimination()
    print("< solution >\n", sol)
    print("< error >\n", error)

    # 2-B
    print("\n2-B. Gaussian elimination with pivotting")
    sol, error = q1.GaussianElimination_Pivotting()
    print("< solution >\n", sol)
    print("< error >\n", error)

    # 2-C
    print("\n2-C. Gauss_Seidel method")
    sol = q1.Gauss_Seidal()
    print("< solution >\n", sol)

    # 3-A
    print("\n[ Considering round-off error ]")
    sol, error = q1.GaussianElimination_roundoff()
    print("3-A. Gaussian Elimination\n", sol)
    print("<error>\n", error)

    # 3-B
    sol, error = q1.GaussianElimination_Pivotting_roundoff()
    print("\n3-B. Gaussian Elimination with Pivotting\n", sol)
    print("<error>\n", error)

    # 3-C
    sol = q1.Gauss_Seidal_roundoff()
    print("\n3-C. Gauss_Seidel method\n", sol)