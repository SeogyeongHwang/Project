## 1번

import numpy as np
import matplotlib.pyplot as plt
import math

class sin_taylor_series:

    def f(self, x):
        return np.sin(x)

    def PN(self, a, delta, N):
        approx_sum = 0
        sign = 1
        x = a + delta
        h = a - x

        for n in range(N+1):
            if (n % 2 == 0):
                approx_sum += sign * np.sin(x) * (h**n) / math.factorial(n)
            elif (n % 2 == 1):
                approx_sum += sign * np.cos(x) * (h**n) / math.factorial(n)
                sign *= -1

        return approx_sum

    def truncation_error(self, x, a, N):

        for n in range(N+1):
            approx_sum = self.PN(x, a, n)

            Et = np.abs((self.f(x) - approx_sum) / self.f(x)) * 100
            print(f'N = {n} --> result : {approx_sum:.4f}, error :{Et:.4f}%')

        return

    def plot(self, delta):
        x = np.linspace(-np.pi/2, np.pi/2, 1000)
        a = np.pi/6 + delta
        P1 = lambda x: np.sin(a) + np.cos(a) * (x - a)
        P2 = lambda x: np.sin(a) + np.cos(a) * (x - a) - (np.sin(a)/2)*(x-a)**2
        P4 = lambda x: np.sin(a) + np.cos(a) * (x - a) - (np.sin(a)/2)*(x-a)**2 - (np.cos(a)/math.factorial(3))*(x-a)**3 + (np.sin(a)/math.factorial(4))*(x-a)**4

        y1 = P1(x)
        y2 = P2(x)
        y4 = P4(x)

        plt.plot(x, y1, label='P1')
        plt.plot(x, y2, label='P2')
        plt.plot(x, y4, label='P3')
        plt.legend()
        plt.grid()
        plt.show()

        return

if __name__ == "__main__":
    q1 = sin_taylor_series()
    # Test the function for sin(x) near pi/6 with N = 7
    x = np.pi / 6 + 0.1  # Choose a point near pi/6
    N = 7  # Degree of the Taylor polynomial
    #pproximation = PN(x, N)
    #print("Approximation of sin(x) with N = 7:", approximation)

    # 1 - 2
    q1.truncation_error(np.pi/6, 0.3, 7)

    # 1 - 3
    q1.plot(0.3)