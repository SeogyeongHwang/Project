# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 12:22:09 2024

@author: Eve
"""
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import math

class SinTaylorSeries:

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

    def truncation_error(self, x, a, N, f):
        for n in range(N+1):
            approx_sum = self.PN(x, a, n)

            Et = np.abs((self.f(x) - approx_sum) / self.f(x)) * 100
            f.write(f'N = {n} --> result : {approx_sum:.4f}, error : {Et:.4f}%\n')
            #print(f'N = {n} --> result : {approx_sum:.4f}, error :{Et:.4f}%')

        return

    def plot(self, delta):
        title = "Plot Taylor Series Result" 
        fig, ax = plt.subplots(figsize=(8, 8))
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
        plt.plot(x, y4, label='P4')
        plt.legend()
        ax.set_title(title, fontsize = 16)
        ax.set_xlabel('x', fontsize = 16)
        ax.set_ylabel('y', fontsize = 16)
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        plt.grid()
        path = os.path.join("Q1_Results", "Plot_taylorSeries_result")
        fig.savefig(path)
        plt.show()

        return

if __name__ == "__main__":
    if not os.path.exists("Q1_Results"):
        os.makedirs("Q1_Results")
        
    q1 = SinTaylorSeries()
    delta = 0.3
    # Test the function for sin(x) near pi/6 with N = 7
    x = np.pi / 6 + delta # Choose a point near pi/6
    N = 7  # Degree of the Taylor polynomial
    
    # 1 - 2
    path = os.path.join("Q1_Results", "truncationError_results.txt")
    with open(path, 'w') as f:
        f.write(" Results for %s\n\n" % datetime.now())
        f.write("< Results for truncation error >\n")
        f.write(f'degree = {N}, delta = {delta}\n\n')
        q1.truncation_error(np.pi/6, delta, N, f)

    # 1 - 3
    q1.plot(delta)