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
$$x = {{ 1.5124959, -1.49743733, 3.53518394, -5.72930549 }}^{T}$$ 
$$error = {{ 5.432 x {10}^{-15}, 5.931 x {10}^{-16}, 1.256 x {10}^{-16}, 7.7512 x {10}^{-16} }}^{T}$$    
    
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
$$x = {{ 1.5124959, -1.49743733, 3.53518394, -5.72930549 }}^{T}$$ 
$$error = {{ 2.2021 x {10}^{-15}, 3.707 x {10}^{-15}, 1.13 x {10}^{-15}, 1.395 x {10}^{-15} }}^{T}$$  


##### C. Gauss-Siedel method    

