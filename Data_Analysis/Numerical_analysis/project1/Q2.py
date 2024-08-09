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


class RootEstimator:

    def __init__(self, a, b, c, eps = 0.001):
        self.a = a
        self.b = b
        self.c = c
        self.eps = eps
        return

    def f(self, x):
        return(x - self.a) * (x - self.b) * (x - self.c)

    def df(self, x):
        # f(x) = x ^ 3 -(a+b+c)* x^2 + (ab+ac+bc)*x - abc
        # chain rule
        return (x - self.a) * (x - self.b) +  (x - self.a) * (x - self.c) + (x - self.b) * (x - self.c)

    def g(self, x):
        d = - (self.a*self.b + self.a*self.c + self.b*self.c) # => x = g(x)
        return ((x - self.a) * (x - self.b) * (x - self.c)  + d*x) /d

    def plot_f_x(self):
        x = np.arange(self.a - 1, self.c + 1.1, 0.01)
        y = [self.f(x_i) for x_i in x]
        title = "Plot for f(x) = (x - %s) * (x - %s) * (x - %s)\n" %(self.a, self.b, self.c) 
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(x, y)
        ax.set_title(title, fontsize = 16)
        ax.set_xlabel('x', fontsize = 16)
        ax.set_ylabel('f(x)', fontsize = 16)
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        plt.grid()
        plt.show()
        
        if not os.path.exists("Q2_Results"):
            os.makedirs("Q2_Results")
        path = os.path.join("Results", "Plot_f(x)=(x-%s)(x-%s)(x-%s)" %(self.a, self.b, self.c))
        fig.savefig(path)
        plt.close()

        return

    def bisection_method(self, xl, xu, f):       
        n_i = 0
        if self.f(xl) * self.f(xu) >= 0:
            f.write("There's no roots\n")
            return

        while xu - xl > self.eps:
            n_i += 1
            xr = (xl + xu) / 2
            D = self.f(xl) * self.f(xr)

            if D <= 0:
                xu = xr
            elif D > 0:
                xl = xr

            Et = np.abs((xu - xl) / xu)

            if n_i >=  2:
                Ea = np.abs((xr - (xl+xu)/2) / xr)
            else: Ea = np.nan

        
            f.write(f'n = {n_i} --> xl : {xl:.4f}, xu: {xu:.4f}, xr: {(xl + xu) / 2:4f}, Ea: {Ea:4f}%, error: {Et:.4f}%\n')

        return

    def fixedPointIteration_method(self, x_old, f):
      n_i = 0
      
      title = "Plot fixedPointIteration Result" 
      fig, ax = plt.subplots(figsize=(8, 8))
      xx = np.arange(self.a - 1, self.c + 1.1, 0.01)
      y1 = xx
      plt.plot(xx, self.g(xx), label="x = g(x)")
      plt.plot(xx, y1, label="y = x")
      plt.legend()

      prev_Et = 100
      while abs(self.g(x_old) - x_old) >= self.eps:
        n_i += 1
        x_new = self.g(x_old)

        Et = np.abs((x_new - x_old) / x_new) * 100
        ratio = Et / prev_Et
        f.write(f'n = {n_i} --> result : {x_new:.4f}, error:{Et:.4f}, ratio = {ratio:.4f}\n')
        prev_Et = Et
        x_old = x_new

        plt.plot(x_new, self.g(x_new), 'm.')
        
      ax.set_title(title, fontsize = 16)
      ax.set_xlabel('x', fontsize = 16)
      ax.set_ylabel('f(x)', fontsize = 16)
      ax.axhline(y=0, color='k')
      ax.axvline(x=0, color='k')
      plt.grid()
      path = os.path.join("Q2_Results", "Plot_fixedPointIteration_result_f(x)=(x-%s)(x-%s)(x-%s)" %(self.a, self.b, self.c))
      fig.savefig(path)
      plt.show()

      return

    def newtonRaphson_method(self, x_old, f):
        n_i = 0 # number of iteration

        title = "Plot newtonRaphson Result" 
        fig, ax = plt.subplots(figsize=(8, 8))
        x = np.arange(self.a - 1, self.c + 1.1, 0.01)
        y = self.f(x)
        plt.plot(x, y)
        plt.grid()

        while abs(self.f(x_old) / self.df(x_old)) >= self.eps:
            n_i += 1

            x_new = x_old - (self.f(x_old) / self.df(x_old))

            plt.scatter(x_old, self.f(x_old))

            Et = np.abs((x_new - x_old) / x_new) * 100
            f.write(f'n = {n_i}, x_new = {x_new:.4f}, Error: {Et :4f}\n')
            x_old = x_new
            
        ax.set_title(title, fontsize = 16)
        ax.set_xlabel('x', fontsize = 16)
        ax.set_ylabel('f(x)', fontsize = 16)
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        path = os.path.join("Q2_Results", "Plot_newtonRaphson_result_f(x)=(x-%s)(x-%s)(x-%s)" %(self.a, self.b, self.c))
        fig.savefig(path)
        plt.show()

        return


if __name__ == "__main__":
    # 2 - 1 and 2 - 2
    a, b, c = 1, 2, 3
    xl = 0.5
    xu = 1.5
    re = RootEstimator(a, b, c)
    re.plot_f_x()
    # bisection_method
    path = os.path.join("Q2_Results", "bisection_results.txt")
    with open(path, 'w') as f:
        f.write(" Results for %s\n\n" % datetime.now())
        f.write("< Results for Bisction Method >\n")
        f.write(f'f(x)=(x-{a})(x-{b})(x-{c})\n\n')
        re.bisection_method(xl, xu, f)
    # fixedPointIteration_method
    path = os.path.join("Q2_Results", "fixedPointIteration_results.txt")
    with open(path, 'w') as f:
        f.write(" Results for %s\n\n" % datetime.now())
        f.write("< Results for fixedPointIteration Method >\n")
        f.write(f'f(x)=(x-{a})(x-{b})(x-{c})\n\n')
        re.fixedPointIteration_method(xl, f)
    # newtonRaphson_method
    path = os.path.join("Q2_Results", "newtonRaphson_results.txt")
    with open(path, 'w') as f:
        f.write(" Results for %s\n\n" % datetime.now())
        f.write("< Results for newtonRaphson Method >\n")
        f.write(f'f(x)=(x-{a})(x-{b})(x-{c})\n\n')
        re.newtonRaphson_method(xl, f)
        
    # 2 - 3
    a, b, c = 1, 20, 21
    xl = 0
    xu = 1.5
    re = RootEstimator(a, b, c)
    re.plot_f_x()
    # bisection_method 
    path = os.path.join("Q2_Results", "bisection_results(2).txt")
    with open(path, 'w') as f:
        f.write(" Results for %s\n\n" % datetime.now())
        f.write("< Results for Bisction Method >\n")
        f.write(f'f(x)=(x-{a})(x-{b})(x-{c})\n\n')
        re.bisection_method(xl, xu, f)
    # fixedPointIteration_method
    path = os.path.join("Q2_Results", "fixedPointIteration_results(2).txt")
    with open(path, 'w') as f:
        f.write(" Results for %s\n\n" % datetime.now())
        f.write("< Results for fixedPointIteration Method >\n")
        f.write(f'f(x)=(x-{a})(x-{b})(x-{c})\n\n')
        re.fixedPointIteration_method(xl, f)
    # newtonRaphson_method
    path = os.path.join("Q2_Results", "newtonRaphson_results(2).txt")
    with open(path, 'w') as f:
        f.write(" Results for %s\n\n" % datetime.now())
        f.write("< Results for newtonRaphson Method >\n")
        f.write(f'f(x)=(x-{a})(x-{b})(x-{c})\n\n')
        re.newtonRaphson_method(xl, f)