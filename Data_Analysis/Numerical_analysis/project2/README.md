# Project 2

### 1. Consider the following liner system equation, where $\delta$ = ${10}^{-n}$, n = 1,2,3,...
     [ δ 3 2 1 ] ( x1 )   ( -3 )
     [ 4 δ 7 5 ] ( x2 ) = (  2 )
     [ 8 2 δ 2 ] ( x3 )   ( -2 ) 
     [ 0 1 2 δ ] ( x4 )   ( -5 )
(The following is the result of substituting δ=0.1)    
    
#### 1) Find the true solution of x = ${{x_{1}, x_{2}, x_{3}, x_{4}}}^{T}$ using matrix inversion
    
If we check def '__init__', 'inversion' in Q2 class, these part summarizes the A and B arrays as follows and finds the x-solution vector through the inversion method. When you run the code, below is the result.

        [ δ 3 2 1 ]       ( -3 )
    A = [ 4 δ 7 5 ] , B = (  2 ) 
        [ 8 2 δ 2 ]       ( -2 )
        [ 0 1 2 δ ]       ( -5 )     

, x = $𝐴_{-1}𝐵$ can be obtained and true x value can be found. Therefore, it was confirmed that x is the following result.    
    
#### 2) Use the following methods to find out the solution, and measure the true relative errors.
##### A. Naïve Gaussian elimination

'NaiveGaussianElimination' function is a code that performs forward and backward and performs Gauss elimination operations, and at the end, the true relative error was also calculated to return.
This method is a method of solving a simultaneous equation after performing a forward substitution and a backward substitution in the form of 'Ux=d'. The matrix of U and d obtained after the forward substitution is output as follows.

        [ 0.1   3    2    1    ]       (   -3  )
    U = [  0 -119.9 -73  -35   ] , d = (  122  )
        [  0    0   -15 -8.525 ]       ( -4.17 )
        [  0    0    0  -0.983 ]       ( 5.631 )

After that, the solution and true relative errors finally obtained while proceeding with the backward are as follows.    
$$x = {[1.5124959, -1.49743733, 3.53518394, -5.72930549]}^{T}$$ 
$$error = {[5.432 x {10}^{-15}, 5.931 x {10}^{-16}, 1.256 x {10}^{-16}, 7.7512 x {10}^{-16}]}^{T}$$    
    
##### B. Gaussian elimination with pivoting    

'GaussianElimination Pivotting' code that adds only the pivoting part of A's Gaussian Elimination code. In order to prevent the order of the equations from affecting the operation results, pivotting was added in the above method. Below is the pivotting result of the matrix connected by A and B.

     [ 8 2 δ 2 -2 ] 
     [ 4 δ 7 5  2 ] 
     [ δ 3 2 1 -3 ] 
     [ 0 1 2 δ  5 ] 

After that, if you output the U and d matrices obtained through the forward substitution...

        [ 8   2    0.1      2   ]       (  -2   )
    U = [ 0 -0.9   6.95     4   ] , d = (   3   )
        [ 0   0  24.9724  14.2  ]       ( 6.942 )
        [ 0   0     0    -0.983 ]       ( 5.631 )

The final solution and true relative error obtained after the backward process are as follows.
$$x = {[1.5124959, -1.49743733, 3.53518394, -5.72930549]}^{T}$$ 
$$error = {[2.2021 x {10}^{-15}, 3.707 x {10}^{-15}, 1.13 x {10}^{-15}, 1.395 x {10}^{-15}]}^{T}$$  


##### C. Gauss-Siedel method    

For the Gauss-Siedel method to converge, the value of the [i, i]th coefficient must be greater than the absolute value of the coefficients in the other same row. However, a given matrix does not satisfy this. So it shows ***"This is not diagonally dominant"*** for the result.    

#### 3) Consider the round-off error during the computation in Problem-2). Rounding operation can be easily implemented by np.round() function.    
Example: np.round(value, decimals = k) ,    (The result when decimal is 4)    

##### A. Naïve Gaussian Elimination    

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

The roundoff error was considered as follows using the np.round method in code 2. The results accordingly are as follows.    
$$x = {[1.511, -1.4974, 3.5352, -5.7293]}^{T}$$ 
$$error = {[9.89 x {10}^{-4}, 2.4926 x {10}^{-5}, 4.544 x {10}^{-6}, 9.58 x {10}^{-7}]}^{T}$$    

##### B. Gaussian Elimination with Pivottnig    

Since the np.round method was used in the same place as in A above, I did not attach the code separately, but the result is as follows.    
$$x = {[1.5125, -1.4973, 3.5352, -5.7293]}^{T}$$ 
$$error = {[2.708 x {10}^{-6}, 9.171 x {10}^{-5}, 4.544 x {10}^{-6}, 9.58 x {10}^{-7}]}^{T}$$    

##### C. Gauss-Seidel method    

The following results are obtained for the same reason as in 2).    
***"This is not diagonally dominant"***    



#### 4) Analyze the results caused by round-off errors for various schemes.    

1. In Näve Gaussian Elimination, which did not take into account the roundoff error, it was confirmed that the final solution was very similar to the true value, so it had a very small true relative error.    
$$x = {[1.5124959, -1.49743733, 3.53518394, -5.72930549]}^{T}$$ 
$$error = {[5.432 x {10}^{-15}, 5.931 x {10}^{-16}, 1.256 x {10}^{-16}, 7.7512 x {10}^{-16}]}^{T}$$    
This indicates that it is numerically stable when rounding is not applied.    
However, if you check the results when considering the round off error, you can see that the solution value changes slightly and the relative error value becomes larger.
$$x = {[1.511, -1.4974, 3.5352, -5.7293]}^{T}$$ 
$$error = {[9.89 x {10}^{-4}, 2.4926 x {10}^{-5}, 4.544 x {10}^{-6}, 9.58 x {10}^{-7}]}^{T}$$       
This suggests that the Naïve Gaussian Elimination method is sensitive to precision.    
    
3. If you check the results of the Gaussian elimination method to which pivotting was applied, you can see that a slightly larger error occurred compared to the one to which pivoting was not applied.
$$x = {[1.5124959, -1.49743733, 3.53518394, -5.72930549]}^{T}$$ 
$$error = {[2.2021 x {10}^{-15}, 3.707 x {10}^{-15}, 1.13 x {10}^{-15}, 1.395 x {10}^{-15}]}^{T}$$    
On the other hand, when comparing the results of applying the roundoff error, one side of pivoting showed a smaller error.    

4. Since $\delta$ = 0.1 is added, the difference between what was pivoted and what was not pivoted is not noticeable, but if a very small $\delta$ is added, Gaussian elimination can diverge if pivoted is not performed.

5. Considering the Roundoff error, it can be seen that larger values of decimal make the solution more accurate and smaller values of true relative errors.    
Unlike above, below is the result of giving decimal a value of 10.
+ A. Naïve Gaussian elimination
  $$x = {[1.5124959, -1.49743733, 3.53518394, -5.72930549]}^{T}$$ 
  $$error = {[4.6 x {10}^{-10}, 3.79 x {10}^{-11}, 6.7054 x {10}^{-11}, 6.5173 x {10}^{-11}]}^{T}$$
+ B. Gaussian elimination with pivoting
  $$x = {[1.5124959, -1.49743733, 3.53518394, -5.72930549]}^{T}$$ 
  $$error = {[6.8875 x {10}^{-11}, 1.624 x {10}^{-10}, 4.61 x {10}^{-11}, 5.7 x {10}^{-11}]}^{T}$$
    
The Gauss-Seidel method does not have convergence, so do not attach the result value separately.    



### 2. The following figures show the Gaussian (normal) distribution function (PDF), and its cumulative distribution function (CDF).    

![Alt_text](https://github.com/SeogyeongHwang/Project/blob/a5f143b66ba3fdbcf2880bc35f508ac15db40c6b/Data_Analysis/Numerical_analysis/project2/Q2_Results/PDF%26CDF.jpg)    

#### 1) Generate the N x 1 data x having the normal distribution using np.random.normal() function with mean=0, standard deviation=4.    

    def normal_distribution(self, N, mean=0, std_dev=4): 
        x = np.random.normal(mean, std_dev, N) 
        return x 

Above method was created to generate data.    

#### 2) Design the function for CDF of **x** , i.e., y = CDF(x) = $F_{x}(x)$ = P[ **x** $\leq$ x]. The probability can be simply obtained by counting the number of elements which satisfies **x** $\leq$ x.    
#### 3) Set N = 100. Pick the random number $x_{i} \in$ (-10, 10), and find $y_{i} = CDF(x_{i}$). Then, draw the scattered plot for ($x_{i}, y_{i}$), | = 1, ..., 15.    
The 'calculate_cdf' method was created so that the value of all elements smaller than x specified by Data was the CDF(x) value, and the CDF_values values were saved in the form of a list in the 'design_CDF' method. After N = 100, the plot graph is as follows.    
![Alt_text](https://github.com/SeogyeongHwang/Project/blob/898aa890a375d8d196415ebb8fb3c58920d8afdd/Data_Analysis/Numerical_analysis/project2/Q2_Results/Scatter%20plot%20of%20CDF%20values.png)    

#### 4) Fit the data ($x_{i}, y_{i}$) using the linear, polynomial, and non-linear regression methods, and measure the coefficient of determination for each method.    
##### A. Linear Regression    
If you find the coefficient of linear regression and plot it, result is below.    
![Alt_text](https://github.com/SeogyeongHwang/Project/blob/898aa890a375d8d196415ebb8fb3c58920d8afdd/Data_Analysis/Numerical_analysis/project2/Q2_Results/Linear%20Regression(N%3D100).png)        

##### B. Polynomial Regression    
If the coefficient of the Polynomial Regression can be obtained and plotted accordingly, the result when degree = 2 is as follows.    
![Alt_text](https://github.com/SeogyeongHwang/Project/blob/898aa890a375d8d196415ebb8fb3c58920d8afdd/Data_Analysis/Numerical_analysis/project2/Q2_Results/Polynomial%20Regression(N%3D100%2C%20degree%3D2).png)     

##### C. Non-linear Regression    
After calculating the coefficient of the nonlineal regression, the log function in the nonlinea has a value only when x is positive, so the matrix is modified so that only the positive x remains and the coefficient is configured to be calculated. Plotting the results through this is as follows.    
![Alt_text](https://github.com/SeogyeongHwang/Project/blob/898aa890a375d8d196415ebb8fb3c58920d8afdd/Data_Analysis/Numerical_analysis/project2/Q2_Results/Non-linear%20Regression(N%3D100).png)       

#### 5) Repeat the problem using various N (e.g. N = 25, 50, 500, 1000, ...)    
Looking at the results plotted by changing N, the results above were N = 100, and the results plotted by changing N are graphs plotting linear, polynomial, and non-linear regression results, respectively.     
##### (1) N = 25,    
<p float="left">
   <img src="https://github.com/SeogyeongHwang/Project/blob/898aa890a375d8d196415ebb8fb3c58920d8afdd/Data_Analysis/Numerical_analysis/project2/Q2_Results/Linear%20Regression(N%3D25).png" width="49%" height="49%">
   <img src="https://github.com/SeogyeongHwang/Project/blob/898aa890a375d8d196415ebb8fb3c58920d8afdd/Data_Analysis/Numerical_analysis/project2/Q2_Results/Polynomial%20Regression(N%3D25%2C%20degree%3D2).png" width="49%" height="49%">
   <img src="https://github.com/SeogyeongHwang/Project/blob/898aa890a375d8d196415ebb8fb3c58920d8afdd/Data_Analysis/Numerical_analysis/project2/Q2_Results/Non-linear%20Regression(N%3D25).png" width="49%" height="49%">
   </p>

##### (2) N = 50    
<p float="left">
   <img src="https://github.com/SeogyeongHwang/Project/blob/898aa890a375d8d196415ebb8fb3c58920d8afdd/Data_Analysis/Numerical_analysis/project2/Q2_Results/Linear%20Regression(N%3D50).png" width="49%" height="49%">
   <img src="https://github.com/SeogyeongHwang/Project/blob/898aa890a375d8d196415ebb8fb3c58920d8afdd/Data_Analysis/Numerical_analysis/project2/Q2_Results/Polynomial%20Regression(N%3D50%2C%20degree%3D2).png" width="49%" height="49%">
   <img src="https://github.com/SeogyeongHwang/Project/blob/898aa890a375d8d196415ebb8fb3c58920d8afdd/Data_Analysis/Numerical_analysis/project2/Q2_Results/Non-linear%20Regression(N%3D50).png" width="49%" height="49%">
   </p>

##### (3) N = 150    
<p float="left">
   <img src="https://github.com/SeogyeongHwang/Project/blob/898aa890a375d8d196415ebb8fb3c58920d8afdd/Data_Analysis/Numerical_analysis/project2/Q2_Results/Linear%20Regression(N%3D150).png" width="49%" height="49%">
   <img src="https://github.com/SeogyeongHwang/Project/blob/898aa890a375d8d196415ebb8fb3c58920d8afdd/Data_Analysis/Numerical_analysis/project2/Q2_Results/Polynomial%20Regression(N%3D150%2C%20degree%3D2).png" width="49%" height="49%">
   <img src="https://github.com/SeogyeongHwang/Project/blob/898aa890a375d8d196415ebb8fb3c58920d8afdd/Data_Analysis/Numerical_analysis/project2/Q2_Results/Non-linear%20Regression(N%3D150).png" width="49%" height="49%">
   </p>

Looking at the graphs that change as the value of N is changed, the number of sample data increases, so the shape of the graph appears more accurately along the points with a little more scatter plot. Looking at the outline alone, assuming that the degree of the polynomial regression is set to 2, it can be seen that the non-linear regression shows only the positive portion of the graph, but the non-linear is the most accurate along the points.    

#### 6) Maximize the regression performance as much as you can.    
In order to increase the regression performance, the degree of the polynomial regression was increased.     
<p float="left">
   <img src="https://github.com/SeogyeongHwang/Project/blob/898aa890a375d8d196415ebb8fb3c58920d8afdd/Data_Analysis/Numerical_analysis/project2/Q2_Results/Polynomial%20Regression(N%3D100%2C%20degree%3D3).png" width="49%" height="49%">
   <img src="https://github.com/SeogyeongHwang/Project/blob/898aa890a375d8d196415ebb8fb3c58920d8afdd/Data_Analysis/Numerical_analysis/project2/Q2_Results/Polynomial%20Regression(N%3D100%2C%20degree%3D6).png" width="49%" height="49%">
   <img src="https://github.com/SeogyeongHwang/Project/blob/898aa890a375d8d196415ebb8fb3c58920d8afdd/Data_Analysis/Numerical_analysis/project2/Q2_Results/Polynomial%20Regression(N%3D100%2C%20degree%3D4).png" width="49%" height="49%">
   <img src="https://github.com/SeogyeongHwang/Project/blob/898aa890a375d8d196415ebb8fb3c58920d8afdd/Data_Analysis/Numerical_analysis/project2/Q2_Results/Polynomial%20Regression(N%3D100%2C%20degree%3D10).png" width="49%" height="49%">
   </p>
    
These figures are in order when degree=3, 6, 4, and 10. As you can see from the graph alone, the higher the degree of degrease, the more accurate the performance of the regression becomes and the line passes through every point.
