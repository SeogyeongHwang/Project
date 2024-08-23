# Project 2

### 1. Consider the following liner system equation, where $\delta$ = ${10}^{-n}$, n = 1,2,3,...
     [ δ 3 2 1 ] ( x1 )   ( -3 )
     [ 4 δ 7 5 ] ( x2 ) = ( 2 )
     [ 8 2 δ 2 ] ( x3 )   ( -2 ) 
     [ 0 1 2 δ ] ( x4 )   ( -5 )
(The following is the result of substituting δ=0.1)    
    
#### 1) Find the true solution of x = ${{x_{1}, x_{2}, x_{3}, x_{4}}}^{T}$ using matrix inversion
    
If we check def '__init__', 'inversion' in Q2 class, these part summarizes the A and B arrays as follows and finds the x-solution vector through the inversion method. When you run the code, below is the result.    
        [ δ 3 2 1 ]       ( -3 )
If  A = [ 4 δ 7 5 ] , B = ( 2 )  , x = $𝐴_{-1}𝐵$ can be obtained and true x value can be found. Therefore, 
        [ 8 2 δ 2 ]       ( -2 ) 
        [ 0 1 2 δ ]       ( -5 )
it was confirmed that x is the following result.  
