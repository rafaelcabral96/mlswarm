# Overview

This package trains neural networks using swarm-like optimization algorithms. 

By flattening the weights of a neural network, network training can be seen as a problem of directly minimizing a multivariate (cost) function. In this framework, particle swarm optimization algorithms can be used to minimize this multivariate function, where each particle will have a set of neural network weights associated with it.

The package also includes a gradient free variant of the optimization algorithm, where no backpropagation is required. Some advantages are listed at the end.

## Relevant article
Diogo A. Gomes (2019). DERIVATIVE FREE OPTIMIZATION USING GAUSSIAN CLOUD ?

## Description
**mlswarm** contains two operable classes:
1. *neuralnet* - train neural networks
2. *function* - minimize functions

For a *neuralnet* object there are three main methods (see examples):
1. nn = neuralnet(...) - define neural network architecture and create neural network
2. nn = init_cloud(N) - Initialize cloud with N particles
3. nn.train(...) - Define the training data, algorithm parameters and start the algorithm

For a *function* object there are three main methods (see examples):
1. func = function(lambda x: ...) - create a function object 
2. func = init_cloud(...) - Define array of initial particle positions
3. func.minimize(...) - Define the algorithms parameters and start the algorithm

There are three available optimization algorithms:
1. *swarm* - swarm-like optimization algorithm
2. *swarm_derivfree* - similar to the former but derivative free, by using gaussian clouds
3. *gradient_descent* - gradient descent optimization

## Examples
Jupyter notebook examples can be found on the [github page](https://github.com/rafaelcabral96/mlswarm) that perform:
1. Minimization of univariate and multivariate functions
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