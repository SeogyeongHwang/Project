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
|<table> <tr><th>Order n</th><th>Section</th><th> </th><th>Approximation root</th><th>Error</th><th> </th></tr><tr><th> </th><th>$X_{l}$</th><th>$X_{u}$</th><th>$X_{r}$</th><th>$ㅣE_{a}ㅣ$ (%)</th><th>$ㅣE_{t}ㅣ$ (%)$</th></tr><tr><td>1</td><td>0.5000</td><td>2.000</td><td>1.250000</td><td>nan</td><td>75.0000</td></tr><tr><td>2</td><td>0.5000</td><td>1.2500</td><td>0.875000</td><td>0.300000</td><td>60.0000</td></tr><tr><td>3</td><td>0.8750</td><td>1.2500</td><td>1.062500</td><td>0.214286</td><td>30.0000</td></tr><tr><td>4</td><td>0.8750</td><td>1.0625</td><td>0.968750</td><td>0.088235</td><td>17.6471</td></tr><tr><td>5</td><td>0.9688</td><td>1.0625</td><td>1.015625</td><td>0.048387</td><td>8.8235</td></tr><tr><td>6</td><td>0.9688</td><td>1.0156</td><td>0.992188</td><td>0.023077</td><td>4.6154</td></tr><tr><td>7</td><td>0.9922</td><td>1.0156</td><td>1.003606</td><td>0.011811</td><td>2.3077</td></tr><tr><td>8</td><td>0.9922</td><td>1.0039</td><td>0.998047</td><td>0.005837</td><td>1.1673</td></tr><tr><td>9</td><td>0.9980</td><td>1.0039</td><td>1.000977</td><td>0.002935</td><td>0.5837</td></tr><tr><td>10</td><td>0.9980</td><td>1.0010</td><td>0.999512</td><td>0.001463</td><td>0.2927</td></tr><tr><td>11</td><td>0.9995</td><td>1.0010</td><td>1.000244</td><td>0.000733</td><td>0.1463</td></tr><tr><td>12</td><td>0.9995</td><td>1.0002</td><td>0.999878</td><td>0.000366</td><td>0.0732</td></tr> </table>| <table>  <tr><th>Order n</th><th>Section</th><th> </th><th>Approximation root</th><th>Error</th><th> </th></tr><tr><th> </th><th>$X_{l}$</th><th>$X_{u}$</th><th>$X_{r}$</th><th>$ㅣE_{a}ㅣ$ (%)</th><th>$ㅣE_{t}ㅣ$ (%)$</th></tr><tr><td>1</td><td>1.5000</td><td>2.0000</td><td>1.750000</td><td> </td><td>0.2500</td></tr><tr><td>2</td><td>1.75000</td><td>2.0000</td><td>1.875000</td><td>0.071429</td><td>0.1250</td></tr><tr><td>3</td><td>1.8750</td><td>2.0000</td><td>1.937500</td><td>0.033333</td><td>0.0625</td></tr><tr><td>4</td><td>1.9375</td><td>2.0000</td><td>1.968750</td><td>0.016129</td><td>0.0312</td></tr><tr><td>5</td><td>1.9688</td><td>2.0000</td><td>1.984375</td><td>0.007937</td><td>0.0156</td></tr><tr><td>6</td><td>1.9844</td><td>2.0000</td><td>1.992188</td><td>0.003937</td><td>0.0078</td></tr><tr><td>7</td><td>1.9922</td><td>2.0000</td><td>1.996094</td><td>0.001961</td><td>0.0039</td></tr><tr><td>8</td><td>1.9961</td><td>2.0000</td><td>1.998047</td><td>0.000978</td><td>0.0020</td></tr><tr><td>9</td><td>1.9980</td><td>2.0000</td><td>1.999023</td><td>0.000489</td><td>0.0010</td></tr><tr><td>10</td><td>1.9990</td><td>2.0000</td><td>1.999512</td><td>0.000244</td><td>0.0005</td></tr> </table>| <table> <tr><th>Order n</th><th>Section</th><th> </th><th>Approximation root</th><th>Error</th><th> </th></tr><tr><th> </th><th>$X_{l}$</th><th>$X_{u}$</th><th>$X_{r}$</th><th>$ㅣE_{a}ㅣ$ (%)</th><th>$ㅣE_{t}ㅣ$ (%)$</th></tr><tr><td>1</td><td>0.5000</td><td>1.0000</td><td>0.750000</td><td> </td><td>0.5000</td></tr><tr><td>2</td><td>0.75000</td><td>1.0000</td><td>0.875000</td><td>0.166667</td><td>0.2500</td></tr><tr><td>3</td><td>0.8750</td><td>1.0000</td><td>0.937500</td><td>0.071429</td><td>0.1250</td></tr><tr><td>4</td><td>0.9375</td><td>1.0000</td><td>0.968750</td><td>0.033333</td><td>0.0625</td></tr><tr><td>5</td><td>0.9688</td><td>1.0000</td><td>0.984375</td><td>0.006129</td><td>0.0312</td></tr><tr><td>6</td><td>0.9844</td><td>1.0000</td><td>0.992188</td><td>0.007937</td><td>0.0156</td></tr><tr><td>7</td><td>0.9922</td><td>1.0000</td><td>0.996094</td><td>0.003937</td><td>0.0078</td></tr><tr><td>8</td><td>0.9961</td><td>1.0000</td><td>0.998047</td><td>0.001961</td><td>0.0039</td></tr><tr><td>9</td><td>0.9980</td><td>1.0000</td><td>0.999023</td><td>0.000978</td><td>0.0020</td></tr><tr><td>10</td><td>0.9990</td><td>1.0000</td><td>0.999512</td><td>0.000489</td><td>0.0010</td></tr></table>|

When the range of the Bisection method was changed and applied, the wider the x range, the more iteration numbers came out, and it was approximated to the closest value among true roots larger than the $X_{l}$ value. When the interval between the sections was 1, (0.5, 1.5) and (1.5, 2.5), the values of the interval and the measurement root were almost the same, and the values (1.5, 2.5) were higher by 1. The values for error were generally similar.

##### B. Simple fixed-point iteration method

|x = 0.5|x = 3.2|
|--|--|
|<table> <tr><th>i</th><th>$X_{i}$</th><th>$ㅣE_{t}ㅣ$ (%)</th><th>$ㅣE_{t}ㅣ_{i} / ㅣE_{t}ㅣ_{i-1}$</th></tr><tr><td>1</td><td>0.6705</td><td>25.4237</td><td>0.2542</td></tr><tr><td>2</td><td>0.7632</td><td>12.1572</td><td>0.4782</td></tr><tr><td>3</td><td>0.8228</td><td>7.2365</td><td>0.5952</td></tr><tr><td>4</td><td>0.8641</td><td>4.7788</td><td>0.6604</td></tr><tr><td>5</td><td>0.8941</td><td>3.3533</td><td>0.7017</td></tr><tr><td>6</td><td>0.9165</td><td>2.4476</td><td>0.7299</td></tr><tr><td>7</td><td>0.9336</td><td>1.8357</td><td>0.7500</td></tr><tr><td>8</td><td>0.9469</td><td>1.4041</td><td>0.7649</td></tr><tr><td>9</td><td>0.9574</td><td>1.0897</td><td>0.7761</td></tr><tr><td>10</td><td>0.9656</td><td>0.8551</td><td>0.7847</td></tr><tr><td>11</td><td>0.9722</td><td>0.6767</td><td>0.7914</td></tr><tr><td>12</td><td>0.9775</td><td>0.5391</td><td>0.7966</td></tr><tr><td>13</td><td>0.9817</td><td>0.4317</td><td>0.8008</td></tr><tr><td>14</td><td>0.9851</td><td>0.3471</td><td>0.8041</td></tr><tr><td>15</td><td>0.9879</td><td>0.2801</td><td>0.8068</td></tr><tr><td>16</td><td>0.9901</td><td>0.2266</td><td>0.8089</td></tr><tr><td>17</td><td>0.9919</td><td>0.1837</td><td>0.8107</td></tr><tr><td>18</td><td>0.9934</td><td>0.1491</td><td>0.8121</td></tr><tr><td>19</td><td>0.9946</td><td>0.1213</td><td>0.8132</td></tr> </table>| <table>  <tr><th>i</th><th>$X_{i}$</th><th>$ㅣE_{t}ㅣ$ (%)</th><th>$ㅣE_{t}ㅣ_{i} / ㅣE_{t}ㅣ_{i-1}$</th></tr><tr><td>1</td><td>3.1520</td><td>1.5228</td><td>0.0152</td></tr><tr><td>2</td><td>3.1177</td><td>1.0988</td><td>0.7215</td></tr><tr><td>3</td><td>3.0924</td><td>0.8193</td><td>0.7457</td></tr><tr><td>4</td><td>3.0732</td><td>0.6248</td><td>0.7626</td></tr><tr><td>5</td><td>3.0584</td><td>0.4841</td><td>0.7749</td></tr><tr><td>6</td><td>3.0468</td><td>0.3796</td><td>0.7841</td></tr><tr><td>7</td><td>3.0377</td><td>0.3003</td><td>0.7911</td></tr><tr><td>8</td><td>3.0305</td><td>0.2392</td><td>0.7965</td></tr><tr><td>9</td><td>3.0247</td><td>0.1916</td><td>0.8008</td></tr><tr><td>10</td><td>3.0200</td><td>0.1540</td><td>0.8042</td></tr><tr><td>11</td><td>3.0163</td><td>0.1243</td><td>0.8069</td></tr><tr><td>12</td><td>3.0132</td><td>0.1006</td><td>0.8090</td></tr><tr><td>13</td><td>3.0108</td><td>0.0815</td><td>0.8107</td></tr><tr><td>14</td><td>3.0088</td><td>0.0662</td><td>0.8121</td></tr><tr><td>15</td><td>3.0072</td><td>0.0538</td><td>0.8133</td></tr><tr><td>16</td><td>3.0059</td><td>0.0438</td><td>0.8142</td></tr><tr><td>17</td><td>3.0048</td><td>0.0357</td><td>0.8149</td></tr> </table>|

<p float="left">
   <img src="https://github.com/SeogyeongHwang/Project/blob/73c4b93ccf6de02c730ea36be80c457e0f187b79/Data_Analysis/Numerical_analysis/project1/Q2_Results/Plot_fixedPointIteration_result_f(x)%3D(x-1)(x-2)(x-3)_(x%3D0.5).png" width="49%" height="49%">
   <img src="https://github.com/SeogyeongHwang/Project/blob/73c4b93ccf6de02c730ea36be80c457e0f187b79/Data_Analysis/Numerical_analysis/project1/Q2_Results/Plot_fixedPointIteration_result_f(x)%3D(x-1)(x-2)(x-3)_(x%3D3.2).png" width="49%" height="49%">
   </p>

When the initial x value is added, it converges to the true root value to find the approximation. However, no matter what x value is added, the value does not converge with 2, but when a value slightly less than 2, it converges to one side, and when a value slightly greater than 2, it converges to three sides.

##### C. Newton-Raphson method

|x = 0.5|x = 2.4|x = 3.5|
|--|--|--|
|<table> <tr><th>i</th><th>$X_{i}$</th><th>$ㅣE_{t}ㅣ$ (%)</th></tr><tr><td>1</td><td>0.8261</td><td>39.473684</td></tr><tr><td>2</td><td>0.9677</td><td>14.633353</td></tr><tr><td>3</td><td>0.9984</td><td>3.089616</td></tr><tr><td>4</td><td>1.000</td><td>0.145279</td></tr> </table>| <table> <tr><th>i</th><th>$X_{i}$</th><th>$ㅣE_{t}ㅣ$ (%)</th></tr><tr><td>1</td><td>1.7538</td><td>36.842105</td></tr><tr><td>2</td><td>2.0365</td><td>13.877562</td></tr><tr><td>3</td><td>1.9999</td><td>1.827788</td></tr> </table>| <table> <tr><th>i</th><th>$X_{i}$</th><th>$ㅣE_{t}ㅣ$ (%)</th></tr><tr><td>1</td><td>3.1739</td><td>10.273973</td></tr><tr><td>2</td><td>3.0323</td><td>4.669907</td></tr><tr><td>3</td><td>3.0015</td><td>1.027874</td></tr><tr><td>4</td><td>3.0000</td><td>0.048426</td></tr> </table>|

<p float="left">
   <img src="https://github.com/SeogyeongHwang/Project/blob/73c4b93ccf6de02c730ea36be80c457e0f187b79/Data_Analysis/Numerical_analysis/project1/Q2_Results/Plot_newtonRaphson_result_f(x)%3D(x-1)(x-2)(x-3)_(x%3D0.5).png" width="49%" height="49%">
   <img src="https://github.com/SeogyeongHwang/Project/blob/73c4b93ccf6de02c730ea36be80c457e0f187b79/Data_Analysis/Numerical_analysis/project1/Q2_Results/Plot_newtonRaphson_result_f(x)%3D(x-1)(x-2)(x-3)_(x%3D2.4).png" width="49%" height="49%">
  <img src="https://github.com/SeogyeongHwang/Project/blob/73c4b93ccf6de02c730ea36be80c457e0f187b79/Data_Analysis/Numerical_analysis/project1/Q2_Results/Plot_newtonRaphson_result_f(x)%3D(x-1)(x-2)(x-3)_(x%3D3.5).png" width="49%" height="49%">
   </p>

It approximates the true root closest to the given value of x

#### 3) Repeat the problem (1) and (2) with different coefficients.

![Alt_text](https://github.com/SeogyeongHwang/Project/blob/73c4b93ccf6de02c730ea36be80c457e0f187b79/Data_Analysis/Numerical_analysis/project1/Q2_Results/Plot_f(x)%3D(x-1)(x-20)(x-21).png)

##### A. Bisection method

