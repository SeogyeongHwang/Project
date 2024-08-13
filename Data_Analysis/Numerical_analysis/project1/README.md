# Project 1

### 1. Let's approximate sinusoidal functions using Taylor series. We can approximate the following functions near x = 0 as bellows. 
  
$$\mathrm{e}^{x} = \displaystyle\sum_{n=0}^{\infty} \frac{{x}^{n}}{n!}$$

$$\cos (x) = \displaystyle\sum_{n=0}^{\infty} \frac{{-1}^{n}{x}^{2n}}{(2n)!}$$

$$\sin (x) = \displaystyle\sum_{n=0}^{\infty} \frac{{-1}^{n}{x}^{2n+1}}{(2n + 1)!}$$

#### 1) Let $P_{N}(x)$ be the approximate Taylor polynomials of degree _N_. Find $P_{N}(x)$ for sin(x) function near $\pi$/6.

When approximating the sine function using Taylor series, Taylor series centered on 0 given above can be used. However, However, when approximating to a specific point such as $\frac{\pi}{6}$, $(x-\frac{\pi}{6})$ aligns the center to $\frac{\pi}{6}$ and reflects the sin and cos values that vary according to the center, a different equation comes out from the above. Therefore, $P_{N}(x)$ as below can be obtained.

$$P_{N}(x) = \displaystyle\sum_{n=0}^{\infty} \frac{{sin}^{(n)}(x)}{n!}{(\frac{\pi}{6}-x)}^{n}$$

$$P_{N}(x) = sin(x) + cos(x)$(\frac{\pi}{6}-x) - \frac{sin(x)}{2!}{(\frac{\pi}{6}-x)}^{2} - \frac{cos(x)}{3!}{(\frac{\pi}{6}-x)}^{3} + ... + \frac{{sin}^{(N)}(x)}{N!}{(\frac{\pi}{6}-x)}^{N}$$

#### 2) Find the approximated values for $P_N(x=\frac{\pi}{6}+\delta)$ and its errors.

  (1) N = 4, $\delta$ = 0.1
  |Order n|Result|Error|
  |:---:|:---:|:---:|
  |0|0.5840|16.7921%|
  |1|0.5028|0.5564%|
  |2|0.4999|0.0275%|
  |3|0.5000|0.0005%|
  |4|0.5000|0.0000%|

  (2) N = 5, $\delta$ = 0.2
  |Order n|Result|Error|
  |:---:|:---:|:---:|
  |0|0.6621|32.4172%|
  |1|0..5122|2.4401%|
  |2|0.4990|0.2083%|
  |3|0.5000|0.0084%|
  |4|0.5000|0.0004%|
  |5|0.5000|0.0000%|

  (3) N = 6, $\delta$ = 0.3
  |Order n|Result|Error|
  |:---:|:---:|:---:|
  |0|0.7368|46.7193%|
  |1|0.5297|5.9441%|
  |2|0.4967|0.6583%|
  |3|0.4998|0.0466%|
  |4|0.5000|0.0029%|
  |5|0.5000|0.0001%|
  |6|0.5000|0.0000%|

+ The truncation error of the approximate value of the function using Taylor series was obtained.
+ The larger Delta, the farthest value from the true root value of 0.5 can be seen when the result, which is the approximation sum value, rotates the repetition statement for the first time, and the resulting start error is large, indicating that it must be repeated several times before making the error zero.

#### 3) Plot $P_{1}(x), P_{2}(x), P_{4}(x)$ for $-\frac{\pi}{2}$ < X < $\frac{\pi}{2} in one figure.
<p float="left">
   <img src="https://github.com/SeogyeongHwang/Project/blob/4557b12f5db7cfb89c3c948e2ea3d83c9f12c8a9/Data_Analysis/Numerical_analysis/project1/Q1_Results/Plot_taylorSeries_result(N%3D4%2Cdelta%3D0.1).png" width="49%" height="49%">
   <img src="https://github.com/SeogyeongHwang/Project/blob/4557b12f5db7cfb89c3c948e2ea3d83c9f12c8a9/Data_Analysis/Numerical_analysis/project1/Q1_Results/Plot_taylorSeries_result(N%3D5%2Cdelta%3D0.2).png" width="49%" height="49%">
   <img src="https://github.com/SeogyeongHwang/Project/blob/4557b12f5db7cfb89c3c948e2ea3d83c9f12c8a9/Data_Analysis/Numerical_analysis/project1/Q1_Results/Plot_taylorSeries_result(N%3D6%2Cdelta%3D0.3).png" width="49%" height="49%">
   </p>

This is the graph when the ∆ is 0.1, 0.2, and 0.3, respectively, in order. The graph shows that the larger the delta, the wider the gap between the graphs.    
P1 is a linear form that can predict the increase and decrease of a function between x ranges given as a first-order approximation. P2 and P4 are the addition of derivatives to further reflect the curvature of the function.

##### 4) Compare and analyze the results.


