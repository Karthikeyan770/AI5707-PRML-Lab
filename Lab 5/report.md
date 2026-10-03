# AI5707 — Pattern Recognition and Machine Learning
## Lab 5: Maximum Likelihood Parameter Estimation

### 1. Objective
The objective of this experiment is to estimate regression parameters using Maximum Likelihood Estimation (MLE) under Gaussian observation noise. The supplied Lab 5 notebook applies MLE to the previous-assignment dataset and to the scikit-learn Diabetes dataset.

The assumed model is

$$t_n=\mathbf{w}^{T}\boldsymbol{\phi}(x_n)+\epsilon_n,\qquad \epsilon_n\sim\mathcal{N}(0,\beta^{-1}).$$

The MLE weight estimate is

$$\boxed{\mathbf{w}_{ML}=(\Phi^T\Phi)^{-1}\Phi^T\mathbf{t}}$$

and the MLE noise variance is

$$\boxed{\sigma^2_{ML}=\frac{1}{N}\sum_{n=1}^{N}(t_n-\hat t_n)^2}.$$

The precision is

$$\boxed{\beta_{ML}=\frac{1}{\sigma^2_{ML}}}.$$

### 2. Background Theory
For independent Gaussian errors,

$$p(\mathbf{t}\mid\Phi,\mathbf{w},\beta)=\prod_{n=1}^{N}\mathcal{N}\left(t_n\mid\mathbf{w}^{T}\boldsymbol{\phi}_n,\beta^{-1}\right).$$

Taking the logarithm gives

$$\ln p=\frac{N}{2}\ln\beta-\frac{N}{2}\ln(2\pi)-\frac{\beta}{2}\sum_{n=1}^{N}\left(t_n-\mathbf{w}^{T}\boldsymbol{\phi}_n\right)^2.$$

Setting the derivative with respect to the weights to zero gives the normal equations

$$\Phi^T\Phi\mathbf{w}_{ML}=\Phi^T\mathbf{t}.$$

Hence,

$$\mathbf{w}_{ML}=(\Phi^T\Phi)^{-1}\Phi^T\mathbf{t}.$$

Differentiation with respect to the precision gives

$$\beta_{ML}=\frac{N}{\sum_{n=1}^{N}(t_n-\hat t_n)^2},$$

so

$$\sigma^2_{ML}=\frac{\mathrm{SSE}}{N}.$$

Under the Gaussian-noise assumption, maximizing likelihood is therefore equivalent to minimizing the squared-error objective.

### 3. Experimental Procedure
For the previous-assignment data, the notebook loads `data/noisy_16.txt`, which contains 10,000 samples. A degree-3 polynomial design matrix is used:

$$\Phi(x)=[1,\ x,\ x^2,\ x^3].$$

The implementation uses NumPy's least-squares solver, rather than explicitly forming the matrix inverse. The fitted model and its residuals are then visualized.

### 4. Results on Previous-Assignment Data
The notebook reports the following MLE results for the degree-3 polynomial model:

$$\mathbf{w}_{ML}=[-29.681917,\,-0.501195,\,0.000044,\,0.000094].$$

$$\mathrm{SSE}=13\,876\,478.144457607.$$

$$\sigma^2_{ML}=1387.6478144457606,$$

$$\beta_{ML}=0.000720643948406613.$$

The design-matrix rank is 4, equal to the number of polynomial coefficients.

The notebook also evaluates the normal-equation solution directly. The maximum absolute difference between the two weight estimates is approximately

$$1.28\times10^{-13},$$

showing numerical agreement to machine precision for this dataset.

### 5. Real-World Data: Diabetes Dataset
The second experiment uses the scikit-learn Diabetes dataset and fits

$$\mathbf{y}=X\mathbf{w}+\boldsymbol{\epsilon},\qquad \boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\sigma^2I).$$

An intercept column is added explicitly, and the data are divided into training and test sets using an 80:20 split with a fixed random state of 42.

The estimated coefficient vector is

$$\mathbf{w}_{ML}=[151.345605,\,37.904021,\,-241.964362,\,542.428759,\,347.703844,\,-931.488846,\,518.062277,\,163.419983,\,275.317902,\,736.198859,\,48.670657]^T.$$

The corresponding estimates are

$$\sigma^2_{ML}=2868.549702835577,$$

$$\beta_{ML}=0.00034860821794772967.$$

The predictive performance reported by the notebook is

$$\text{Train RMSE}=53.55884336723094,$$

$$\text{Test RMSE}=53.85344583676589,$$

$$R^2=0.45260276297192015.$$

The notebook also generates an actual-versus-predicted plot and a residual plot for the test set.

### 6. Observations
The experiment confirms the connection between probabilistic Gaussian regression and least-squares estimation. For the previous-assignment data, the normal-equation result matches the numerical least-squares result to machine precision.

For the Diabetes dataset, the training and test RMSE values are close. The reported test $R^2$ is about 0.453, while residual variation remains substantial.

### 7. Computational Complexity
For $N$ samples and $d$ model parameters:

- Forming $X^TX$: $O(Nd^2)$
- Solving the parameter system: approximately $O(d^3)$
- Prediction: $O(Nd)$
- Storing the design matrix: $O(Nd)$

Although the analytical formula contains $(X^TX)^{-1}$, explicitly calculating an inverse is generally less numerically stable than solving the linear system or using a least-squares routine.

### 8. Conclusion
This experiment demonstrates Maximum Likelihood Estimation for linear regression with Gaussian observation noise. The weight estimate is obtained from the normal equations, while the MLE noise variance is obtained from the residual sum of squares divided by the number of observations.

The experiments on both the previous-assignment dataset and the real Diabetes dataset show how MLE provides a probabilistic interpretation of regression while leading to the familiar least-squares solution.


**Q4. What does $\beta$ represent?**  
$\beta$ is the precision of the Gaussian noise model, equal to the reciprocal of the variance.

**Q5. What does $\sigma^2I$ imply?**  
It assumes equal noise variance for the observations and zero covariance between different observations.
