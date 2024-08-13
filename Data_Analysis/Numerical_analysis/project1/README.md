# Project 1

### 1. Let's approximate sinusoidal functions using Taylor series. We can approximate the following functions near x = 0 as bellows. 
  
$$\mathrm{e}^{x} = \displaystyle\sum_{n=0}^{\infty} \frac{{x}^{n}}{n!}$$

$$\cos (x) = \displaystyle\sum_{n=0}^{\infty} \frac{{-1}^{n}{x}^{2n}}{(2n)!}$$

$$\sin (x) = \displaystyle\sum_{n=0}^{\infty} \frac{{-1}^{n}{x}^{2n+1}}{(2n + 1)!}$$

#### 1) Let $P_{N}(x)$ be the approximate Taylor polynomials of degree _N_. Find $P_{N}(x)$ for sin(x) function near $\pi$/6.

When approximating the sine function using Taylor series, Taylor series centered on 0 given above can be used. However, However, when approximating to a specific point such as $\frac{\pi}{6}$, (x-$\frac{\pi}{6}$) aligns the center to $\frac{\pi}{6}$ and reflects the sin and cos values that vary according to the center, a different equation comes out from the above. Therefore, $P_{N}(x)$ as below can be obtained.
$$P_{N}(x) = \displaystyle\sum_{n=0}^{\infty} \frac{{sin}^{(n)}(x)}{n!}{(\frac{\pi}{6}-x)}^{n}$$
