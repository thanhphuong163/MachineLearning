# Linear Regression

## Introduction

Linear regression is a regression method. This method is considered as a **supervised machine learning** model since the fed data must consists of input features and output target. It performs regression tasks in order to find the **linear relationship** between **independent features** and **target**. Therefore, the use of linear regression mostly fall into two categories including forecasting (or prediction) and variational explaination.

The mathematical form of linear regression simply describes the linear dependence of input variables (features) and output variable (target). Given $\boldsymbol{x} = (x_1, \dots , x_m)$ are independent variables, $y$ is denpendent variable, and $\boldsymbol{\theta} = (\theta_0, \dots, \theta_m)$ are coefficients. The model is as the following
$$
y = \theta_0 + \theta_1 x_1 + \dots + \theta_m x_m + \epsilon = \boldsymbol{\overline{x}}^\top \boldsymbol{\theta} + \epsilon,
$$
where $\epsilon$ is a random error, and $\boldsymbol{\overline{x}} = (1, x_1, \dots, x_m)$.

## Data Modeling

Let $X = ({\boldsymbol{\overline{x}}^{(i)}}^\top)_{i=1}^{n}$ be the input data, and $\boldsymbol{y} = (y^{(i)})_{i=1}^{n}$ be the targets, then the data model can be written as
$$
X \boldsymbol{\theta} \approx \boldsymbol{y}.
$$
If we call $\hat{\boldsymbol{y}} = X \boldsymbol{\theta}$ are the predictions, then $\epsilon = \boldsymbol{y} - \hat{\boldsymbol{y}}$. Our goal is to find $\boldsymbol{\theta}$ that minimize the squared error $\epsilon^2$. Now, our problem reduces to an **opimization problem**. Thus, we need to define a **loss function** (or an **objective function**, according to optimization terms) with respect to $\boldsymbol{\theta}$. And the loss function is defined as

$$
\begin{aligned}
\mathcal{L}(\boldsymbol{\theta}) & = \frac{1}{2} \sum_{i=1}^{n}{(y^{(i)} - \hat{y}^{(i)})^2} \\
& = \frac{1}{2} \sum_{i=1}^{n}{(y^{(i)} - {\boldsymbol{\overline{x}}^{(i)}}^\top \boldsymbol{\theta})^2} \\
& = \frac{1}{2} (\boldsymbol{y} - X \boldsymbol{\theta})^\top (\boldsymbol{y} - X \boldsymbol{\theta}).
\end{aligned}
$$

So far, the readers must be wondering "Why is this function? Why is squared error?". Because the error can be negative infinity, and that would be a huge error. Then "How about absoluted error?", this is not a smooth function, which means that it is not differentiable everywhere. Namely, the absoluted error is not differentiable at the optimal point $0$. Furthermore, the loss function comes more naturally when we consider linear regression in terms of **probablistic modeling** (which will be explained later).

## Optimal Solution

### Least Squared Problem

Now, our loss function is a convex quadratic function, and we want to find its minimum. And fortunately, this loss function has only one minimum. One of the most popular way is solving differential equation equal to zero.

$$
\begin{aligned}
\frac{\partial \mathcal{L}(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} & = \boldsymbol{0} \\
X^\top ( \boldsymbol{y} - X \boldsymbol{\theta}) & = \boldsymbol{0} \\
X^\top \boldsymbol{y} & = X^\top X \boldsymbol{\theta}.
\end{aligned}
$$

Set $X^\top \boldsymbol{y} = \boldsymbol{b}$ and $X^\top X = A$, then

$$
A \boldsymbol{\theta} = \boldsymbol{b}.
$$

This equation has the form of linear system of equations. If the readers are familiar with linear algebra, then the solution is $\boldsymbol{\theta} = A^{-1} b$ when $A$ is invertible (non-singular). However, this is not the end of our story. What if $A$ is not invertible (singular). We still can solve this equation with pseudo-inverse of $A$, denoted $A^{\dagger}$, which is itself obtained by calculating [SVD](https://www2.math.uconn.edu/~leykekhman/courses/MATH3795/Lectures/Lecture_9_Linear_least_squares_SVD.pdf) of $A$.

$$
\begin{aligned}
\boldsymbol{\theta} & = A^{\dagger} b \\
& = (X^\top X)^\dagger X^\top \boldsymbol{y}.
\end{aligned}
$$

The readers can read more about least squared problem and SVD in my [note](fdshk).

### Gradient Descent

Gradient Descent is an optimization algorithm. The idea of Gradient Descent is updating $\boldsymbol{\theta}$ after computing gradient of the gradient of loss function,
$\frac{\Delta \mathcal{L}(\boldsymbol{\theta})}{\Delta \boldsymbol{\theta}} = X^\top ( \boldsymbol{y} - X \boldsymbol{\theta})$. Namely, in each loop, $\boldsymbol{\theta}$ is updated as the following
$$
\boldsymbol{\theta} := \boldsymbol{\theta} - \alpha \frac{\Delta \mathcal{L}(\boldsymbol{\theta})}{\Delta \boldsymbol{\theta}},
$$
where $\alpha$ is a learning rate (usually $\alpha = 10^{-4}$). If you want to know about Gradient Descent, see my [note](fdsa).

## A Probabilistic Interpretation


