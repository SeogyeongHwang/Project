# 2번

import numpy as np
import matplotlib.pyplot as plt
import math

class Q2:
  def normal_distribution(self, N, mean=0, std_dev=4):
    x = np.random.normal(mean, std_dev, N)
    return x

  def calculate_cdf(self, data, x_value):
    sum = 0
    for x in data:
      if x <= x_value:
        sum += 1
    value = sum / len(data)

    return value

  def design_CDF(self, N):
    data = self.normal_distribution(N)
    nums = np.random.uniform(-10, 10, 15)
    cdf_values = []
    for x in nums:
      cdf_values.append(self.calculate_cdf(data, x))

    return nums, cdf_values

  def linear_regression(self, x, y):
    y = np.array(y)

    n = np.size(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x*y)
    sum_x2 = np.sum(x**2)
    sum_y2 = np.sum(y**2)

    a1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    a0 = (sum_y - a1 * sum_x) / n
    r = (n * sum_xy - sum_x * sum_y) / (np.sqrt(n ** sum_x2 - sum_x**2) * np.sqrt(n * sum_y2))

    return a1, a0, r

  def polynomial_regression(self, x, y, degree):
    y = np.array(y)

    n = x.size
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x*y)
    sum_x2 = np.sum(x**2)

    z = np.ones((n, 1))
    for i in range(1, degree + 1):
      z = np.concatenate([z, x.reshape(n, 1)**i], axis = 1)

    a_vec = np.linalg.inv(z.T @ z) @ z.T @ y
    y_predic = z @ a_vec
    r = np.sqrt(1 - (np.sum((y - y_predic)**2) / np.sum((y - np.mean(y))**2)))

    return a_vec, r

  def NonLinear_regression(self, x, y):
    y = np.array(y)

    positive_mask = x > 0
    x = x[positive_mask]
    y = y[positive_mask]

    n = np.size(np.log(x))
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_logxy = np.sum(np.log(x) * np.log(y))
    sum_logx2 = np.sum(np.log(x)**2)
    sum_logy2 = np.sum(np.log(y)**2)
    sum_logx = np.sum(np.log(x))
    sum_logy = np.sum(np.log(y))

    a1 = (n * sum_logxy - sum_logx * sum_logy) / (n * sum_logx2 - sum_logx**2)
    a0 = (sum_logy - a1 * sum_logx) / n
    r = (n * sum_logxy - sum_logx * sum_logy) / np.sqrt((n * sum_logx2 - sum_logx**2)*(n * sum_logy2 - sum_logy**2))

    return a1, a0, r

if __name__ == "__main__":
    N = 100
    q2 = Q2()

    # 1
    x = q2.normal_distribution(N)

    # 2, 3
    x, y = q2.design_CDF(N)
    plt.title("Scatter plot of CDF values")
    plt.scatter(x, y)
    plt.show()

    # 4 - Linear Regression
    a1, a0, r = q2.linear_regression(x, y)

    X = np.linspace(-10, 10)
    Y = a1 * X + a0
    plt.title("Linear Regression")
    plt.plot(X, Y)
    plt.scatter(x, y)
    plt.show()

    # 4 - Polynomial regression
    degree = 2
    p, r = q2.polynomial_regression(x, y, degree)

    X = np.linspace(-10, 10)
    Y = np.zeros_like(X)
    for i in range(degree+1):
      Y += p[i] * (X**i)
    plt.title("Polynomial Regression")
    plt.plot(X, Y)
    plt.scatter(x, y)
    plt.show()

    # 4 - Non linear Regression
    a1, a0, r = q2.NonLinear_regression(x, y)

    X = np.linspace(-10, 10)
    Y = np.exp(a0) * (X ** a1)
    plt.title("Non-linear Regression")
    plt.scatter(x, y)
    plt.plot(X, Y)
    plt.show()