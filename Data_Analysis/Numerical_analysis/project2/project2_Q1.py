# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 12:22:09 2024

@author: Eve
"""
from datetime import datetime
import os
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

  def NaiveGaussianElimination(self, f):
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
    
    f.write("\n< solution >\n")
    f.write(np.array2string(x))
    f.write("\n< error >\n")
    f.write(np.array2string(error))

    return

  def GaussianElimination_Pivotting(self, f):
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

    f.write("\n< solution >\n")
    f.write(np.array2string(x))
    f.write("\n< error >\n")
    f.write(np.array2string(error))

    return

  def Gauss_Seidal(self, f, max_iter = 100, eps = 0.01):
    # check convergence for Gauss-Seidel method
    for i in range(len(self.b)):
      sum_row = 0
      for j in range(len(self.A)):
        if (j != i):
           sum_row += np.abs(self.A[i][j])
      if np.abs(self.A[i][i]) < sum_row:
        f.write("This is not diagonally dominant")
        return

    n = np.size(self.b)
    x_old = np.zeros((n, 1))
    x_new = np.zeros((n, 1))

    for iter in range(max_iter):
      for i in range(n):
        x_new[i] = (self.b[i] - ((self.A[i, :i] @ x_new[:i]) + (self.A[i, i+1:] @ x_old[i+1:]))) / self.A[i, i]

      x_old = x_new.copy()

      if np.sum((self.A @ x_new - self.b.T)**2)**0.5 < eps or iter == max_iter-1:
          f.write(np.array2string(x_new))
          return
      
    f.write(np.array2string(x_new))
    
    return

  def GaussianElimination_roundoff(self, f, decimals=10):
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
    
    f.write("\n< solution >\n")
    f.write(np.array2string(x))
    f.write("\n< error >\n")
    f.write(np.array2string(error))

    return

  def GaussianElimination_Pivotting_roundoff(self, f, decimals=10):
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

    f.write("\n< solution >\n")
    f.write(np.array2string(x))
    f.write("\n< error >\n")
    f.write(np.array2string(error))

    return

  def Gauss_Seidal_roundoff(self, f, max_iter = 100, eps = 0.01, decimals=10):
    # check convergence for Gauss-Seidel method
    for i in range(len(self.b)):
      sum_row = 0
      for j in range(len(self.A)):
        if (j != i):
           sum_row += np.abs(self.A[i][j])
      if np.abs(self.A[i][i]) < sum_row:
          f.write("This is not diagonally dominant")
          return

    n = np.size(self.b)
    x_old = np.zeros((n, 1))
    x_new = np.zeros((n, 1))

    for iter in range(max_iter):
      for i in range(n):
        x_new[i] = (self.b[i] - ((self.A[i, :i] @ x_new[:i]) + (self.A[i, i+1:] @ x_old[i+1:]))) / self.A[i, i]
        x_new[i] = np.round(x_new, decimals = decimals)

      x_old = x_new.copy()

      if np.sum((self.A @ x_new - self.b.T)**2)**0.5 < eps or iter == max_iter-1:
          f.write(np.array2string(x_new))
          return
      
    f.write(np.array2string(x_new))
      
    return

if __name__ == "__main__":        
    n = 1
    lamb = 10**(-n)
    q1 = find_matrix_solution(lamb)
    
    path = os.path.join("Q1_Results.txt")
    with open(path, 'w') as f:
        f.write(" Results for %s\n\n" % datetime.now())
        f.write("< True Solutions and errors >\n")
        f.write(f'when lambda = {lamb}\n\n')
        
        # 1. using matrix inversion
        f.write("1) matrix inversion\n")
        result = q1.inversion()
        f.write(np.array2string(result))
        
        # 2-A. Naive Gaussian elimination
        f.write("\n\n2) Naive Gaussian Elimination")
        q1.NaiveGaussianElimination(f)

        # 2-B. Gaussian elimination with pivotting
        f.write("\n\n3) Gaussian Elimination with Pivotting")
        q1.GaussianElimination_Pivotting(f)

        # 2-C. Gauss Seidal method
        f.write("\n\n4) Gauss Seidal method\n")
        q1.Gauss_Seidal(f)        

        # 3-A. Round-off Error
        f.write("\n\n5) Considering round-off error")
        q1.GaussianElimination_roundoff(f)

        # 3-B. Gaussian Elimination with Pivotting & Roundoff-Error
        f.write("\n\n6) Gaussian ELimination with Pivotting and Round-off Error")
        q1.GaussianElimination_Pivotting_roundoff(f)

        # 3-C. Gauss Seidal Roundoff Error
        f.write("\n\n7) Gauss Seidal Round-off Error\n")
        q1.Gauss_Seidal_roundoff(f)