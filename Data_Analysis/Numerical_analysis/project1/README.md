# Project 1

### 1. Let's approximate sinusoidal functions using Taylor series. We can approximate the following functions near x = 0 as bellows. 
  
$$\mathrm{e}^{x} = \displaystyle\sum_{n=0}^{\infty} \frac{{x}^{n}}{n!}$$

$$\cos (x) = \displaystyle\sum_{n=0}^{\infty} \frac{{-1}^{n}{x}^{2n}}{(2n)!}$$

$$\sin (x) = \displaystyle\sum_{n=0}^{\infty} \frac{{-1}^{n}{x}^{2n+1}}{(2n + 1)!}$$

#### 1) Let $P_{N}(x)$ be the approximate Taylor polynomials of degree _N_. Find $P_{N}(x)$ for sin(x) function near $\pi$/6.

When approximating the sine function using Taylor series, Taylor series centered on 0 given above can be used. However, However, when approximating to a specific point such as $\frac{\pi}{6}$, $(x-\frac{\pi}{6})$ aligns the center to $\frac{\pi}{6}$ and reflects the sin and cos values that vary according to the center, a different equation comes out from the above. Therefore, $P_{N}(x)$ as below can be obtained.

$$P_{N}(x) = \displaystyle\sum_{n=0}^{\infty} \frac{{sin}^{(n)}(x)}{n!}{(\frac{\pi}{6}-x)}^{n}$$

$$P_{N}(x) = sin(x) + cos(x)(\frac{\pi}{6}-x) - \frac{sin(x)}{2!}{(\frac{\pi}{6}-x)}^{2} - \frac{cos(x)}{3!}{(\frac{\pi}{6}-x)}^{3} + ... + \frac{{sin}^{(N)}(x)}{N!}{(\frac{\pi}{6}-x)}^{N}$$

#### 2) Find the approximated values for $P_N(x=\frac{\pi}{6}+\delta)$ and its errors.

|N = 4, $\delta$ = 0.1|N = 5, $\delta$ = 0.2|N = 6, $\delta$ = 0.3|
|--|--|--|
|<table> <tr><th>Order n</th><th>Result</th><th>Error</th></tr><tr><td>0</td><td>0.5840</td><td>16.7921%</td></tr><tr><td>1</td><td>0.5028</td><td>0.5564%</td></tr><tr><td>2</td><td>0.4999</td><td>0.0275%</td></tr><tr><td>3</td><td>0.5000</td><td>0.0005%</td></tr><tr><td>4</td><td>0.5000</td><td>0.0000%</td></tr> </table>| <table> <tr><th>Order n</th><th>Result</th><th>Error</th></tr><tr><td>0</td><td>0.6621</td><td>32.4172%</td></tr><tr><td>1</td><td>0.5122</td><td>2.4401%</td></tr><tr><td>2</td><td>0.4990</td><td>0.2083%</td></tr><tr><td>3</td><td>0.5000</td><td>0.0084%</td></tr><tr><td>4</td><td>0.5000</td><td>0.0004%</td></tr><tr><td>5</td><td>0.5000</td><td>0.0000%</td></tr> </table>| <table> <tr><th>Order n</th><th>Result</th><th>Error</th></tr><tr><td>0</td><td>0.7368</td><td>46.7193%</td></tr><tr><td>1</td><td>0.5297</td><td>5.9441%</td></tr><tr><td>2</td><td>0.4967</td><td>0.6583%</td></tr><tr><td>3</td><td>0.4998</td><td>0.0466%</td></tr><tr><td>4</td><td>0.5000</td><td>0.0029%</td></tr><tr><td>5</td><td>0.5000</td><td>0.0001%</td></tr><tr><td>6</td><td>0.5000</td><td>0.0000%</td></tr> </table>|


+ The truncation error of the approximate value of the function using Taylor series was obtained.
+ The larger delta, the farthest value from the true root value of 0.5 can be seen when the result, which is the approximation sum value, rotates the repetition statement for the first time, and the resulting start error is large, indicating that it must be repeated several times before making the error zero.

#### 3) Plot $P_{1}(x), P_{2}(x), P_{4}(x)$ for $-\frac{\pi}{2}$ < X < $\frac{\pi}{2}$ in one figure.
<p float="left">
   <img src="https://github.com/SeogyeongHwang/Project/blob/4557b12f5db7cfb89c3c948e2ea3d83c9f12c8a9/Data_Analysis/Numerical_analysis/project1/Q1_Results/Plot_taylorSeries_result(N%3D4%2Cdelta%3D0.1).png" width="49%" height="49%">
   <img src="https://github.com/SeogyeongHwang/Project/blob/4557b12f5db7cfb89c3c948e2ea3d83c9f12c8a9/Data_Analysis/Numerical_analysis/project1/Q1_Results/Plot_taylorSeries_result(N%3D5%2Cdelta%3D0.2).png" width="49%" height="49%">
   <img src="https://github.com/SeogyeongHwang/Project/blob/4557b12f5db7cfb89c3c948e2ea3d83c9f12c8a9/Data_Analysis/Numerical_analysis/project1/Q1_Results/Plot_taylorSeries_result(N%3D6%2Cdelta%3D0.3).png" width="49%" height="49%">
   </p>

This is the graph when the $\delta$ is 0.1, 0.2, and 0.3, respectively, in order. The graph shows that the larger the delta, the wider the gap between the graphs.    
P1 is a linear form that can predict the increase and decrease of a function between x ranges given as a first-order approximation. P2 and P4 are the addition of derivatives to further reflect the curvature of the function.

#### 4) Compare and analyze the results.

I organized each methods like below.

      (1) Defined a class called SinTaylorSeries and created the necessary method within the class.
      (2) Defining f(x) = sin(x) in 'f' method.
      (3) The 'PN' method is defined to calculate and return the $P_{n}(x)$ sum in the sin(x) function by taking factors of a, delta, and N.
      (4) Defined the 'truncation_error' method and took the factors of x, a, and N to calculate the error according to the approximation of $P_{n}(x)$. Go around the repetition statement and output the approximation and error rate.
      (5) The plot function determined the expressions of P1, P2, and P4 given in the problem and plots the graph.

In conclusion, looking at the approximation and error output by differently outputting each N and delta in No. 2) above, the delta is large, and the result, which is the 'approx._sum' value, shows gradually different values at 0.5 which is sin(π/6), and the resulting error is also large, requiring a larger n of Pn(x) to reduce the error.    
In the graph above, P1(x) is a linear approximation when sin(x) is $x=\frac{\pi}{6}$. Near $x=\frac{\pi}{6}$, it is close and accurate to sin(x), but if x is far from the center point, it becomes farther from sin(x) and the accuracy decreases. In addition, as N increases, a curved graph appears and becomes closer to sin(x) than when N is 1, and a more accurate approximation is made even in the range of x where x is far from the center point. As you can see from the graph, delta increases and the distance x from the center point widens.    


### 2. Consider the following polynomials.

$$f(x) = (x - a)(x - b)(x - c)$$

#### 1) Let a = 1, b = 2, c = 3. Draw y = f(x) graph.

![Alt_text](https://github.com/SeogyeongHwang/Project/blob/7c4f65930e0defb7b9dbcdb6bd15d3cfe9eb2f3a/Data_Analysis/Numerical_analysis/project1/Q2_Results/Plot_f(x)%3D(x-1)(x-2)(x-3).png)

#### 2) Use the following methods to estimate the roots. Measure the number of iterations, and relative errors for each method.

##### A. Bisection method

|between (0.5, 3.5)|between (1.5, 2.5)|between (0.5, 1.5)|
|--|--|--|
|<table> <tr><th>Order n</th><th>Section</th><th> </th><th>Approximation root</th><th>Error</th><th> </th></tr><tr><th> </th><th>$X_{l}$</th><th>$X_{u}$</th><th>$X_{r}$</th><th>ㅣ$E_{a}$ㅣ(%)</th><th>ㅣ$E_{t}$ㅣ(%)$</th></tr><tr><td>1</td><td>0.5000</td><td>2.000</td><td>1.250000</td><td>nan</td><td>75.0000</td></tr><tr><td>2</td><td>0.5000</td><td>1.2500</td><td>0.875000</td><td>0.300000</td><td>60.0000</td></tr><tr><td>3</td><td>0.8750</td><td>1.2500</td><td>1.062500</td><td>0.214286</td><td>30.0000</td></tr><tr><td>4</td><td>0.8750</td><td>1.0625</td><td>0.968750</td><td>0.088235</td><td>17.6471</td></tr><tr><td>5</td><td>0.9688</td><td>1.0625</td><td>1.015625</td><td>0.048387</td><td>8.8235</td></tr><tr><td>6</td><td>0.9688</td><td>1.0156</td><td>0.992188</td><td>0.023077</td><td>4.6154</td></tr><tr><td>7</td><td>0.9922</td><td>1.0156</td><td>1.003606</td><td>0.011811</td><td>2.3077</td></tr><tr><td>8</td><td>0.9922</td><td>1.0039</td><td>0.998047</td><td>0.005837</td><td>1.1673</td></tr><tr><td>9</td><td>0.9980</td><td>1.0039</td><td>1.000977</td><td>0.002935</td><td>0.5837</td></tr><tr><td>10</td><td>0.9980</td><td>1.0010</td><td>0.999512</td><td>0.001463</td><td>0.2927</td></tr><tr><td>11</td><td>0.9995</td><td>1.0010</td><td>1.000244</td><td>0.000733</td><td>0.1463</td></tr><tr><td>12</td><td>0.9995</td><td>1.0002</td><td>0.999878</td><td>0.000366</td><td>0.0732</td></tr> </table>| <table>  <tr><th>Order n</th><th>Section</th><th> </th><th>Approximation root</th><th>Error</th><th> </th></tr><tr><th> </th><th>$X_{l}$</th><th>$X_{u}$</th><th>$X_{r}$</th><th>ㅣ$E_{a}$ㅣ(%)</th><th>ㅣ$E_{t}$ㅣ(%)$</th></tr><tr><td>1</td><td>1.5000</td><td>2.0000</td><td>1.750000</td><td> </td><td>0.2500</td></tr><tr><td>2</td><td>1.75000</td><td>2.0000</td><td>1.875000</td><td>0.071429</td><td>0.1250</td></tr><tr><td>3</td><td>1.8750</td><td>2.0000</td><td>1.937500</td><td>0.033333</td><td>0.0625</td></tr><tr><td>4</td><td>1.9375</td><td>2.0000</td><td>1.968750</td><td>0.016129</td><td>0.0312</td></tr><tr><td>5</td><td>1.9688</td><td>2.0000</td><td>1.984375</td><td>0.007937</td><td>0.0156</td></tr><tr><td>6</td><td>1.9844</td><td>2.0000</td><td>1.992188</td><td>0.003937</td><td>0.0078</td></tr><tr><td>7</td><td>1.9922</td><td>2.0000</td><td>1.996094</td><td>0.001961</td><td>0.0039</td></tr><tr><td>8</td><td>1.9961</td><td>2.0000</td><td>1.998047</td><td>0.000978</td><td>0.0020</td></tr><tr><td>9</td><td>1.9980</td><td>2.0000/td><td>1.999023</td><td>0.000489</td><td>0.0010</td></tr><tr><td>10</td><td>1.9990</td><td>2.0000</td><td>1.999512</td><td>0.000244</td><td>0.0005</td></tr> </table>| <table> <tr><th>Order n</th><th>Section</th><th> </th><th>Approximation root</th><th>Error</th><th> </th></tr><tr><th> </th><th>$X_{l}$</th><th>$X_{u}$</th><th>$X_{r}$</th><th>ㅣ$E_{a}$ㅣ(%)</th><th>ㅣ$E_{t}$ㅣ(%)$</th></tr><tr><td>1</td><td>0.5000</td><td>1.0000</td><td>0.750000</td><td> </td><td>0.5000</td></tr><tr><td>2</td><td>0.75000</td><td>1.0000</td><td>0.875000</td><td>0.166667</td><td>0.2500</td></tr><tr><td>3</td><td>0.8750</td><td>1.0000</td><td>0.937500</td><td>0.071429</td><td>0.1250</td></tr><tr><td>4</td><td>0.9375</td><td>1.0000</td><td>0.968750</td><td>0.033333</td><td>0.0625</td></tr><tr><td>5</td><td>0.9688</td><td>1.0000</td><td>0.984375</td><td>0.006129</td><td>0.0312</td></tr><tr><td>6</td><td>0.9844</td><td>1.0000</td><td>0.992188</td><td>0.007937</td><td>0.0156</td></tr><tr><td>7</td><td>0.9922</td><td>1.0000</td><td>0.996094</td><td>0.003937</td><td>0.0078</td></tr><tr><td>8</td><td>0.9961</td><td>1.0000</td><td>0.998047</td><td>0.001961</td><td>0.0039</td></tr><tr><td>9</td><td>0.9980</td><td>1.0000/td><td>0.999023</td><td>0.000978</td><td>0.0020</td></tr><tr><td>10</td><td>0.9990</td><td>1.0000</td><td>0.999512</td><td>0.000489</td><td>0.0010</td></tr></table>|


When the range of the Bisection method was changed and applied, the wider the x range, the more iteration numbers came out, and it was approximated to the closest value among true roots larger than the $X_{l}$ value. When the interval between the sections was 1, (0.5, 1.5) and (1.5, 2.5), the values of the interval and the measurement root were almost the same, and the values (1.5, 2.5) were higher by 1. The values for error were generally similar.


##### B. Simple fixed-point iteration method

