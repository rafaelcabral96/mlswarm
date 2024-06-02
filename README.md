# Overview

mlswarm is a robust package designed to train neural networks and minimize functions through innovative swarm-like optimization algorithms. It features a unique gradient-free variant that eliminates the need for backpropagation, showcasing significant computational efficiency and flexibility advantages.

Our algorithm has been benchmarked against traditional methods like Nelder-Mead, Differential Evolution, Mesh Search, and Simulated Annealing. It demonstrated superior performance, achieving optimal solutions with **10x to 50x fewer function evaluations**.

## Installation

Install mlswarm easily using pip:

```
pip install mlswarm
```

## Main idea

Unlike traditional gradient descent which uses a single path, swarm-like optimization algorithms employ a group of "particles" that explore the parameter space collectively. 

Auckley function          |  Particle swarm            | Gradient descent
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/rafaelcabral96/mlswarm/blob/master/Images/plot1.png)  |  ![](https://github.com/rafaelcabral96/mlswarm/blob/master/Images/plot2.png) |  ![](https://github.com/rafaelcabral96/mlswarm/blob/master/Images/plot3.png)

This method is particularly adept at handling non-convex functions where gradient descent may fail by getting stuck in local minima. The second plot shows the cloud of 25 points on the top right and its evolution through the iterations until reaching the minimum at the origin. The third plot shows the results using gradient descent (only one particle), where the optimization was stuck on a local minimum.

The core strength of our approach lies in the particles' ability to "communicate" and share information, significantly enhancing the optimization process.   We replace the problem of minimizing $f$ by the equivalent problem of minimizing $F[m]=\int_{\mathbb{R}^{d}} f(x) m(x) d x$, where $m$ is a Gaussian measure. For better results, we consider the equivalent problem of minimizing:

$$
\min _{m \in \mathcal{P}\left(\mathbb{R}^{d}\right)} F[m]+\gamma G[m]+\beta H[m],
$$

where $G[m] = \int_{\mathbb{R}^{d}} \int_{\mathbb{R}^{d}} \frac{|x-y|^{2}}{2} m(x) m(y) d x d y$ is an attractor term that promotes aggregation (particles kept together) and $H[m] = \int_{\mathbb{R}^{d}} m(x) \ln m(x) d x$ is an entropy term that promotes parameter exploration (particles are repelled). 

In practice we consider a discrete measure $m_0=\frac{1}{N} \sum_{i} \delta_{x_i}$, where each $x_i$ is a particle. The previous equations simplify, and we get the Euler scheme:

$$
x_{k+1}^i=x_k^i-\eta ( F_m (x_k^i)+ \gamma G_m(x_k^i)+ \beta H_m(x_k^i))
$$

There is also an algorithm implementation based on Nesterov's accelerated method.

By flattening the weights of a neural network, network training can be seen as a problem of directly minimizing a multivariate (cost) function. Then, particle swarm optimization algorithms can be used to minimize this multivariate function, where each particle will have a set of neural network weights associated with it.

## Relevant article

Gomes, Alexandra A., and Diogo A. Gomes. "Derivative-Free Global Minimization in One Dimension: Relaxation, Monte Carlo, and Sampling." arXiv preprint (2023).

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
2. *gradient_free* - similar to the former but derivative-free
3. *gradient_descent* - gradient descent optimization

There are four ways of updating the particle cloud:
1. *euler* - new_cloud = old_cloud - dt * gradient
2. *euler_adaptive* - same as *euler* but with adaptive step size (dt)
3. *nesterov* - nesterov update
4. *nesterov_adaptive* - Nesterov update with adaptive restart


## Examples
Jupyter notebook examples can be found on the [github page](https://github.com/rafaelcabral96/mlswarm) that perform:
1. Minimization of univariate and multivariate non-convex functions
2. Linear Regression
3. Logistic Regression
4. Binary classification with 4-Layer Neural Network
5. Binary classification with 4-Layer Neural Network using step activation functions

