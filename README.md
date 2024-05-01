# Overview

This package trains neural networks and minimizes functions using swarm-like optimization algorithms. 

By flattening the weights of a neural network, network training can be seen as a problem of directly minimizing a multivariate (cost) function. In this framework, particle swarm optimization algorithms can be used to minimize this multivariate function, where each particle will have the set of neural network weights associated with it.

The package also includes a gradient free variant of the optimization algorithm, where no backpropagation is required. Some advantages are listed at the end.

## Installation

The package can be installed by:

```
pip install mlswarm
```

## Main idea

Swarm-like optimization algorithms differ from the traditional gradient descent in that they a cloud of "particles" that moves through the parameter space.

Auckley function          |  Particle swarm            | Gradient descent
:-------------------------:|:-------------------------:|:-------------------------:
![]([https://...Dark.png](https://github.com/rafaelcabral96/mlswarm/blob/master/Images/plot1.png))  |  ![]([https://...Ocean.png](https://github.com/rafaelcabral96/mlswarm/blob/master/Images/plot2.png)) |  ![]([https://...Ocean.png](https://github.com/rafaelcabral96/mlswarm/blob/master/Images/plot3.png))

It is particularly effective in optimizing non-convex functions. The second plot shows the cloud of 25 points on the top right and it's evolution trough the iterations until reaching the minimum at the origin. The third plot shows the results using gradient descent (only one particle), where the optimization was stuck on a local minumum.

It's ability to overcome local minumuns is due to the fact that particles "communicate" with each other.  We replace the problem of minimizing $f$ by the equivalent problem of minimizing
$F[m]=\int_{\mathbb{R}^{d}} f(x) m(x) d x$, where $m$ is a Gaussian measure. for better results, we consider the equivalent problem of minimizing:

$$
\min _{m \in \mathcal{P}\left(\mathbb{R}^{d}\right)} F[m]+\gamma G[m]+\beta H[m],
$$

where $G[m] = \int_{\mathbb{R}^{d}} \int_{\mathbb{R}^{d}} \frac{|x-y|^{2}}{2} m(x) m(y) d x d y$ is an attractor term that promotes aggregation (particles kept togueter) and $H[m] = \int_{\mathbb{R}^{d}} m(x) \ln m(x) d x$ is an entropy term that promotes parameter exploration (particles are repelled). 

In practice we consider a discrete measure $m_0=\frac{1}{N} \sum_{i=1}^N \delta_{x_i}$, where each $x_i$ is a particle. The previous equations simplify and we get the Euler squeme:
$$
x_{k+1}^{i}=x_{k}^{i}-\eta\left(F_{m}\left(x_{k}^{i}\right)+\gamma G_{m}\left(x_{k}^{i}\right)+\betaP_{m}\left(x_{k}^{i}\right)\right)
$$
There is also an algorithm implementation based on Nesterov’s accelerated method.

We compared our algorithm with Nelder-Mead, Differential Evolution, Mesh Search, and Simulated Annealing, and it reaches the minmum in about 10x-50x less function evaluations.  

## Relevant article

Gomes, Alexandra A., and Diogo A. Gomes. "Derivative-Free Global Minimization in One Dimension: Relaxation, Monte Carlo, and Sampling." arXiv preprint arXiv:2308.09050 (2023).

## Main Functions
**mlswarm** contains two operable classes:
1. *neuralnet* - train neural networks
2. *function* - minimize functions

For a *function* object there are three main methods (see examples):
1. func = function(lambda x: ...) - create a function object 
2. func = init_cloud(...) - Define array of initial particle positions
3. func.minimize(...) - Define the algorithm's parameters and start the algorithm

For a *neuralnet* object there are three main methods (see examples):
1. nn = neuralnet(...) - define neural network architecture and create neural network
2. nn = init_cloud(N) - Initialize cloud with N particles
3. nn.train(...) - Define the training data, algorithm parameter's and start the algorithm

There are three available optimization algorithms:
1. *gradient* - swarm-like optimization algorithm
2. *gradient_free* - similar to the former but derivative free, by using gaussian clouds
3. *gradient_descent* - gradient descent optimization

There are four ways of updating the particle cloud:
1. *euler* - new_cloud = old_cloud - dt * gradient
2. *euler_adaptive* - same as *euler* but with adaptive step size (dt)
3. *nesterov* - nesterov update
4. *nesterov_adaptive* - nesterov update with adaptive restart


## Examples
Jupyter notebook examples can be found on the [github page](https://github.com/rafaelcabral96/mlswarm) that perform:
1. Minimization of univariate and multivariate non-convex functions
2. Linear Regression
3. Logistic Regression
4. Binary classification with 4-Layer Neural Network
5. Binary classification with 4-Layer Neural Network using step activation functions

## Advantages of the swarm-like gradient free optimization
1. It is possible to use linear and non-differentiable activation functions because there is no backpropagation -  [reference](https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/)

2. (Multivariate) linear regression, which uses linear activation function, can be done (see Example 1.)

3. Steps function can be used for binary classification (see Example 5.). Steps activation functions are probably the simplest and fasted to compute.

4. Since there is no backpropagation, we do not have to worry with the gradients collapsing or exploding, therefore there is more freedom when defining the initial values of the weights. I should try other initialization schemes.

5. Usually the mean of the cloud has a higher validation metric on the test set then the individual particles. I think that by training different particles and using the mean of those particles, we are diminishing the problem of overfitting, hence the better results.

6. So far, better than other derivative-free methods when training Neural Networks -> Generic Algorithms, Simulated Annealing, hill climb, random hill climb available the package Python *mlrose*
