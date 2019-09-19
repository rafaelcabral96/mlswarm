from mlswarm.neural_networks import init_layers, full_forward_propagation, get_accuracy_value
from mlswarm.utils import get_mean, get_var, get_all_parameters_util, kernel_a_finder, flatten_weights, convert_prob_into_class, get_parameters, get_parameters_nn
from mlswarm.train import train_nn, train_nn2, f_minimize
from mlswarm.plots import plot_cost_history_util, plot_cost_mean_history_util, plot_var_history_util, plot_gradS_history_util, integral_on_gaussian_measure_util, plot_function_util, plot_cloud_history_util, plot_cost_history2_util, plot_everything_util 

import os
import numpy as np


class _swarm:

    def __init__(self):

        #necessary variables for training
        self.name = "undefined"
        self.N = None
        self.cloud = None
        self.cloud_mean = None
        self.cloud_var = None

        #training metrics
        self.elapsed_iterations = 0
        self.train_method = None
        self.train_parameters = None
        self.function_evaluations = None
        self.clock_time = None
        self.cpu_time = None

        #history
        self.cost_history = []
        self.cost_history_mean = []
        self.cloud_var_history = [] 
        self.cloud_history = []
        self.cloud_history_mean = []

    def clear(self):
        self.N = None
        self.cloud = None
        self.cloud_mean = None
        self.cloud_var = None
        self.train_method = None
        self.train_parameters = None 
        self.function_evaluations = None
        self.clock_time = None
        self.cpu_time = None 
        self.cost_history = None
        self.cost_history_mean = None 
        self.cloud_var_history = None 
        self.cloud_history = None
        self.cloud_history_mean = None
        self.elapsed_iterations = None


    def plot_cost_history(self, save = False, file_name = "", log = False):
        plot_cost_history_util(self, save, file_name, log)

    def plot_cost_mean_history(self, save = False, file_name = "", log = False):
        plot_cost_mean_history_util(self, save, file_name, log)

    def plot_var_history(self, save = False, file_name = "", log = False):
        plot_var_history_util(self, save, file_name, log)

    def plot_gradS_history(self, save = False, file_name = "", log = False):
        plot_gradS_history_util(self, save, file_name, log)


class neuralnet(_swarm):

    def __init__(self, architecture):
        super().__init__()
        self.architecture = architecture
        self.num_variables = sum( [layer["input_dim"] * layer["output_dim"] + layer["output_dim"] for layer in architecture])
        self.gradS_history = False
        self.best_cost = np.inf
        self.best_nn = None
    
    def init_cloud(self, N, cloud_type = "spread", seed = 42, dispersion_factor = 6, seed_between = 43, dispersion_factor_between = 0):
        self.N = N

        if cloud_type == "spread": 
            self.cloud = [init_layers(self.architecture, seed + i , dispersion_factor, 0, 0) for i in range(N)]
        elif cloud_type == "localized":
            self.cloud = [init_layers(self.architecture, seed, dispersion_factor, seed_between + i, dispersion_factor_between) for i in range(N)]
        else:
            raise Exception("Cloud initialization type not found")

        self.cloud_mean = get_mean(self.cloud)
        cloudf, _, _ = flatten_weights(self.cloud, self.N)
        cloud_var = get_var(cloudf,np.mean(cloudf,axis=0)) #np.var(cloudf, axis=0) works best
        self.cloud_var = np.mean(cloud_var)


    def train(self, X, Y, parameters_dic = {}):

        #get parameters from parameters_dic
        parameters = get_parameters_nn(self, X, Y, parameters_dic)

        #find optimal kernel_a
        #if kernel_a == "auto":
        #cloudf, _, _ = flatten_weights(self.cloud, self.N) 
        #kernel_a = kernel_a_finder(cloudf, self.N)

        #start training
        train_nn(*parameters)
        #train_nn2(self, X.T, Y.T, "swarm_derivfree", max_epochs = 300, n_batches = 1, batch_size = X.shape[0], learning_rate=0.5, cost_type = 'error_classification', kernel_a = 0.1,  alpha_init = 0, alpha_rate = 5, beta = 0, gamma = 0, verbose = True, var_epsilon = 0)

    def forward_propagation(self, X, cloud):
        Y, _ = full_forward_propagation(X.T, cloud, self.architecture)
        return Y.T

    def prediction_accuracy_particle(self, X_test, Y_test, acc_type, particle=0):
        Y_test_hat = self.forward_propagation(X_test, self.cloud[particle])
        acc_test = get_accuracy_value(Y_test_hat, Y_test, acc_type)
        print("Test set accuracy using neuralnet {:}: {:.5f}".format(particle, acc_test))

    def prediction_accuracy_mean_particle(self, X_test, Y_test, acc_type):
        Y_test_hat = self.forward_propagation(X_test, self.cloud_mean)
        acc_test = get_accuracy_value(Y_test_hat, Y_test, acc_type)
        print("Test set accuracy using cloud mean: {:.5f}".format(acc_test))

    def prediction_accuracy_best_particle(self, X_test, Y_test, acc_type):
        Y_test_hat = self.forward_propagation(X_test, self.best_nn)
        acc_test = get_accuracy_value(Y_test_hat, Y_test, acc_type)
        print("Test set accuracy using best neuralnet: {:.5f}".format(acc_test))


    def set_cloud(self, cloud):
        self.cloud = cloud
        self.cloud_mean = get_mean(self.cloud)
        self.cloud_var  = get_var(self.cloud)

    def get_cloud_flattened(self):
        cloudf, _, _ = flatten_weights(self.cloud, self.N) 
        return cloudf

    def convert_to_class(Y_hat):
        return convert_prob_into_class(Y_hat)

    def clear(self):
        super.clear(self)
        self.num_variables = None
        self.architecture = None



class function(_swarm):

    def __init__(self, func, name = ""):

        """Create function object. Necessary before initializing cloud and doing optimization.

        Parameters:
        func (function): Function to minimize. Look at example to see syntax.
        
        name (string, default: ""): Function Name. 

        Example:

        # create function object - f(x,y) = x^2 + y^2
        f = lambda x: x[0]**2 + x[1]**2
        func = function(f, "Quadratic Function")
        
        #initialize cloud consisting of two particles in positions (-1,-1) and (2,2)
        func.init_cloud([[-1,-1],[2,2]])

        #Start optimization algorithm with default hyperparameters
        func.minimize()

        """

        super().__init__()
        self.func = func
        self.name = name
        self.function_values = None
        self.cloud_mean_func = None
        self.best_value = None
        self.best_position = None
        self.gradS_history = []

    def init_cloud(self, cloud):
        """Initialize cloud of particles. Necessary before doing numerical optimization.

        Parameters:
        cloud (numpy.array): matrix containing particle positions where each row is a particle and collum is a spatial dimensions
        
        Example:

        #initialize cloud consisting of two particles in positions -1 and 2
        func.init_cloud([[-1],[2]])

        """

        #Particles positions. To be updated during optimization
        self.cloud = np.array(cloud).astype(float)
        #Store image of initial cloud
        self.cloud_0 = self.cloud + 0

        #Compute cloud properties
        self.N = len(self.cloud)
        self.cloud_mean = np.mean(self.cloud, axis = 0)
        self.cloud_var = np.mean(get_var(self.cloud))
        self.function_values = self.func(self.cloud.T)
        
        #store best value respective position found. To be updated during optimization
        self.best_value = np.nanmin(self.function_values)
        self.best_position = self.cloud[np.nanargmin(self.function_values)]
        
    def evaluate(self, cloud):

        """Evaluate function values on given cloud.

        Parameters:
        cloud (numpy.array): matrix containing particle positions where each row is a particle and column is a spatial dimensions
        
        Example:

        # create function object - f(x) = x^2
        f = lambda x: x**2
        func = function(f, "Quadratic Function")
        
        #evaluate x^2 on positions -1 and 2
        func.evaluate([[-1],[2]])

        """

        return self.func(cloud).T


    def minimize(self, parameters_dic = {}):

        """Optimize function acording to hyperparameters specified in parameters_dic.

        Parameters:
        parameters_dic (dic): Dictionary containing hyperparameters. See below.

        parameters_dic:

        max_iterations (int, default: 10000): Maximum amount of iteration.  
        epsilon (float, default:0): Convergence achieved when change in the function value at cloud mean is below epsilon.
        var_epsilon (float, default:0.0001): Convergence achieved when cloud variance is below var_epsilon.
        learning_rate (float, default:0.01): Learning rate of step size.
        method  (string, default: 'gradient_free'): Optimization method - one of - 'gradient' (using swarm algorithm with gradient info), 'gradient_free' (using swarm algorithm without gradient info), 'gradient_descent' (traditional gradient descent).
        algorithm  (string, default: 'euler_adaptive'): Optimizer - one of - 'euler', 'nesterov', 'euler_adaptive', 'nesterov_adaptive'.
        restart_cloud  (bol, default: False): Restart cloud if convergence position is far away from best position found. New cloud is centered on minimum found and has a 9 time smaller variance.
        kernel_a (float, default: 0.01): Constant in kernel function. Lower values leads kernel to be more 'spread' and thus particles comunicate more with each other.
        alpha_init (float, default: 0): Initial value of alpha constant. Associated with aggregation term that brings particles toguether.
        alpha_rate (float, default: 1): Rate of increase of alpha.
        beta (float, default: 1): Entropy term. Associated with term that brings particles apart and promotes search. Only relevant when using 'gradient' method.
        gamma_init (float, default: 0): Initial value of gamma constant. Associated with term that promotes cloud to have a gaussian structure.
        gamma_rate (float, default: 1): Rate of increase of gamma.
        verbose (bol, default: False): Print basic cloud information in each iteration.
        track_history = d.get('track_history', True): Stores cloud history information necessary to do some plots. Slows down optimization.
        
        """

        #get parameters from parameters_dic
        parameters = get_parameters(self, parameters_dic)

        #start minimization
        f_minimize(*parameters)
        
        #Print final results
        print("Cloud mean: " + str(self.cloud_mean) )
        print("Function value at cloud mean: " + str(self.cloud_mean_func))
        print("Function value evaluated {:01} times".format(self.function_evaluations))
        print("Best Particle at position: " + str(self.best_position) + " with function value: " + str(self.best_value))

        #restart cloud at lowest position
        restart_cloud = parameters[6]
        if restart_cloud:

            cond1 = (self.cloud_mean_func > self.best_value)
            cond2 = (np.abs(self.best_value - self.cloud_mean_func) > 0.001)

            if cond1 & cond2:

                print("\n----------------------------------------//------------------------------------")
                print("Cloud converged far away from lowest function value found")
                print("Restarting cloud centered on lowest function value")

                new_cloud = ((self.cloud_0 - np.mean(self.cloud_0, axis=0) )/3  + self.best_position.T)
                new_cloud_var = np.mean(get_var(new_cloud))

                cond3 = (new_cloud_var > var_epsilon)
                if not cond3:
                    print("New cloud variance below var_epsilon")
                    return

                self.init_cloud(new_cloud.T)
                alpha_init += 0
                self.minimize(max_iterations, var_epsilon, learning_rate, 
                              method, restart_cloud, kernel_a, alpha_init, 
                              alpha_rate, beta, gamma_init, gamma_rate, verbose, track_history)


    def get_all_parameters(self,save = False, file_name = ""):

        """Get list with all optimization results. Use after optimization.

        Parameters:
        save (bool, default: False): Save results in csv file with name given by file_name
        file_name (string, dafault: ''): File name 

        Returns:
        header, data (list, list): header - list describing results; data - list containing results

        """

        get_all_parameters_util(self, save, file_name)


    def plot_everything(self, folder_name, f_dim, log = False, limits =[-5,5,-5,5]):

        """Plot all diagnostic plots - Function, cloud history, cost history, variance, etc. Available for 1D and 2D functions. Only use this method if optimization was done with 'track_history' hyperparameter set to true. Plots saved in '/Images/folder_name'.


        Parameters:
        folder_name (string): Name of folder to save plots. If does not exist a new folder will be created.
        f_dim (int): Number of function dimensions - 1 or 2. 
        log (bool, default: False): Also include diagnostic plots with an y-axis using logarithmic scale.
        limits (list, default: [-5,5,-5,5]): Plot limits for cloud history, cost history and function plots. limits = [x_min,x_max,y_min,y_max]

        """

        plot_everything_util(self, folder_name, f_dim, log, limits)


    def integral_on_gaussian_measure(self, plot = False, save = False, file_name = ""):
        integral_on_gaussian_measure_util(self, plot, save, file_name)

    def plot_function(self, save = False, file_name = "", dimension2 = False, limits =[-5,5,-5,5], X = None, Y = None, z = None):
        plot_function_util(self, save, file_name, dimension2, limits, X, Y, z)

    def plot_cloud_history(self, save = False, file_name = "", dimension2 = False, limits = [-5,5,-5,5], X = None, Y = None, z = None, log = False):
        plot_cloud_history_util(self, save, file_name, dimension2, limits, X, Y, z, log)

    def plot_cost_history2(self, save = False, file_name = "", log = False):
        plot_cost_history2_util(self, save, file_name, log)


def plot_data(x, fx, title, xlabel, ylabel, legend = None, log = False, save = False, file_name = ""):
    plot_data_util(x, fx, title, xlabel, ylabel, legend, log, save, file_name)