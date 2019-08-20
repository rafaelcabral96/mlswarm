from mlswarm.neural_networks import init_layers, full_forward_propagation, get_accuracy_value
from mlswarm.utils import get_mean, get_var, plot_matrix, plot_list, plot_2D_trajectory, plot_xy, plot_2D_func, get_2D_plot_data, kernel_a_finder, flatten_weights, convert_prob_into_class, integral_on_gaussian_measure_util
from mlswarm.train import train_nn, train_fmin

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
        self.cost_history = None
        self.cost_history_mean = None 
        self.cloud_var_history = None 
        self.cloud_history = None
        self.cloud_history_mean = None

    #def plot_function(self, save):

    def plot_cost_history(self, save = False, file_name = "", log = False):

        if not "gradient_descent" in self.train_method: 
            if not log:
                plot_matrix(self.cost_history, self.cost_history_mean, 'Function Value', 'Iteration t', '$f(x_i(t))$', r'$f(\bar{x}(t))$', save, file_name)  
            else:
                plot_matrix(np.absolute(self.cost_history), np.absolute(self.cost_history_mean), 'Function Value', 'Iteration t', '$|f(x_i(t))|$', r'$|f(\bar{x}(t))|$', save, file_name, True)                                                   
        else:
            if not log:
                plot_list(self.cost_history, 'Function Value',  'Iteration t',  r'$f(x(t))$', save,  file_name)
            else:
                plot_list(np.absolute(self.cost_history), 'Function Value',  'Iteration t',  r'$|f(x(t))|$', save,  file_name, True)

    def plot_cost_mean_history(self, save = False, file_name = "", log = False):
        if not log:
            plot_list(np.mean(self.cost_history,axis=1), "Average function value", "Iteration t",  r'$\sum_{i=1}^N f(x_i(t))/N$', save, file_name) 
        else:
            plot_list(np.mean(self.cost_history,axis=1), "Average function value", "Iteration t",  r'$|\sum_{i=1}^N f(x_i(t))/N|$', save, file_name, True) 

    def plot_var_history(self, save = False, file_name = "", log = False):
        if not log:
            plot_list(self.cloud_var_history,'Cloud variance','Iteration t', r'$\sigma (t)$', save, file_name) 
        else:
            plot_list(self.cloud_var_history,'Cloud variance','Iteration t', r'$\sigma (t)$', save, file_name, True) 

    def plot_gradS_history(self, save = False, file_name = "", log = False):
        if not log:
            plot_list(self.gradS_history, r"$||\nabla S||$" , "Iteration t", r"$||\nabla S|| (t)$", save, file_name)     
        else:
            plot_list(self.gradS_history, r"$||\nabla S||$" , "Iteration t", r"$||\nabla S|| (t)$", save, file_name, True)     

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


class neuralnet(_swarm):

    def __init__(self, architecture):
        super().__init__()
        self.architecture = architecture
        self.num_variables = sum( [layer["input_dim"] * layer["output_dim"] + layer["output_dim"] for layer in architecture])
        self.gradS_history = False
    
    def init_cloud(self, cloud_type = "spread", seed = 42, dispersion_factor = 6, seed_between = 43, dispersion_factor_between = 0):
        self.N = N

        if cloud_type == "spread": 
            self.cloud = [init_layers(self.architecture, seed + i , dispersion_factor, 0, 0) for i in range(N)]
        elif cloud_type == "localized":
            self.cloud = [init_layers(self.architecture, seed, dispersion_factor, seed_between + i, dispersion_factor_between) for i in range(N)]
        else:
            raise Exception("Cloud initialization type not found")

        self.cloud_mean = get_mean(self.cloud)


    def train(self, X, Y, method, max_epochs, n_batches, batch_size,
              learning_rate, cost_type, kernel_a = "auto",  alpha_init = 0,
               alpha_rate = 1, beta = 0, gamma = 1, verbose = True, var_epsilon = 0):

        if (method == "sgd") and (self.N > 1): 
            self.init_cloud(1)

        #transposing input and output data
        X = X.T
        Y = Y.T

        #find optimal kernel_a
        if kernel_a == "auto":
            cloudf, _, _ = flatten_weights(self.cloud, self.N) 
            kernel_a = kernel_a_finder(cloudf, self.N)

        self.train_method = method
        self.cloud, self.cloud_mean, self.cloud_var, self.cost_history, self.cost_history_mean =  train_nn(X, Y, self.cloud, self.architecture, method, max_epochs, n_batches,  batch_size, learning_rate, cost_type, self.N, kernel_a, alpha_init, alpha_rate, beta, gamma, verbose, var_epsilon)

    def forward_propagation(self, X, cloud):
        Y, _ = full_forward_propagation(X.T, cloud, self.architecture)
        return Y.T

    def prediction_accuracy_particle(self, X_test, Y_test, acc_type, particle=0):
        Y_test_hat = self.forward_propagation(X_test, self.cloud[particle])
        acc_test = get_accuracy_value(Y_test_hat, Y_test, acc_type)
        print("Test set accuracy using particle {:}: {:.5f}".format(particle, acc_test))

    def prediction_accuracy_mean_particle(self, X_test, Y_test, acc_type):
        Y_test_hat = self.forward_propagation(X_test, self.cloud_mean)
        acc_test = get_accuracy_value(Y_test_hat, Y_test, acc_type)
        print("Test set accuracy using cloud mean: {:.5f}".format(acc_test))

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


def search_cloud(self, search_space, N, function_dim):
    sample = 10000
    #grid = np.array([[-5,-5,-5],[5,5,5]])
    dist = np.random.uniform(search_space[0], search_space[1], (sample, function_dim)).T
    func_values = self.func(dist)
    center = dist.take(np.argmin(func_values), axis=1)
    return np.random.multivariate_normal(center,np.identity(function_dim),N,)

class function(_swarm):

    def __init__(self, func, name = "", expression = ""):
        super().__init__()
        self.func = func
        self.name = name
        self.function_values = None
        self.cloud_mean_func = None

    def init_cloud(self, cloud, auto = False, search_space = None, N = None , function_dim = None):
        if auto:
            self.cloud = search_cloud(self, search_space, N, function_dim)
        else:
            self.cloud = cloud.T
        self.N = len(self.cloud)
        self.cloud_mean = np.mean(self.cloud, axis = 0)
        self.cloud_var = np.mean(get_var(self.cloud))

    def minimize(self, max_iterations, var_epsilon, learning_rate, 
         method = "swarm_derivfree", N_sample = False, kernel_a = "auto", alpha_init = 0, 
         alpha_rate = 1, beta = 0, gamma_init = 1, gamma_rate = 0, verbose= False, track_history = True):

        #number of particles
        if "gradient_descent" in method: 
            self.N = 1

        #find optimal kernel_a
        if kernel_a == "auto":
            kernel_a = kernel_a_finder(self.cloud, self.N)

        self.train_method = method
        self.train_parameters = [max_iterations, var_epsilon, learning_rate, kernel_a, alpha_init, alpha_rate, beta, gamma_init, gamma_rate]
      
        train_fmin(self, max_iterations, var_epsilon, learning_rate, method, N_sample, kernel_a, alpha_init, alpha_rate, beta, gamma_init, gamma_rate, verbose, track_history)

    def evaluate(self, cloud):
        return self.func(cloud.T).T

    def get_all_parameters(self,save = False, file_name = ""):
        
        header = ["name", "N", "cloud_var", "cloud_mean" , "cloud_mean_func" ,"train_method", "elapsed_iterations", "function_evaluations", "clock_time", "cpu_time", "max_iterations", "var_epsilon", "learning_rate", "kernel_a", "alpha_init", "alpha_rate", "beta", "gamma", "gamma_rate"]
        data = [self.name, self.N, self.cloud_var,np.array2string(self.cloud_mean, precision = 5), np.format_float_scientific(self.cloud_mean_func.flatten()) ,self.train_method, self.elapsed_iterations, self.function_evaluations, self.clock_time, self.cpu_time]
        data = data + self.train_parameters
    
        if save:
            import pandas 
            df = pandas.DataFrame([data], columns = header)
            df.to_csv(file_name, sep=',', index=False)
        else:
            return header, data

    #only for univariate functions
    def integral_on_gaussian_measure(self, plot = False, save = False, file_name = ""):
        self.integral_history = integral_on_gaussian_measure_util(self)
        if plot:
            plot_list(self.integral_history, 'Function integral on gaussian measure',  'Iteration t',  r'$\int f(x,t) m(x,t) dx$', save,  file_name)

    def plot_function(self, save = False, file_name = "", dimension2 = False, limits =[-5,5,-5,5], X = None, Y = None, z = None):
        if not dimension2:
            x = np.array([np.linspace(limits[0], limits[1], 2000)])
            fx = self.func(x)
            plot_xy(x.flatten(), [fx.flatten()], 'Function', 'x',  r'$f(x)$',None, False, save, file_name)
        else:
            if X is None:
                X, Y, z  = get_2D_plot_data(self, limits)
            plot_2D_func(X, Y, z, save, file_name)

    def plot_cloud_history(self, save = False, file_name = "", dimension2 = False, limits = [-5,5,-5,5], X = None, Y = None, z = None, log = False):
            
        if dimension2:
            if X is None:
                X, Y, z  = get_2D_plot_data(self, limits)

        if not "gradient_descent" in self.train_method: 
            if not dimension2:
                cloud_history = [it[0] for it in self.cloud_history]
                if not log:
                    plot_matrix(cloud_history, self.cloud_history_mean, 'Particle positions',  'Iteration t', r'$x_i(t)$', r'$\bar{x} (t)$', save,  file_name)
                else:
                    plot_matrix(np.absolute(cloud_history), np.absolute(self.cloud_history_mean), 'Particle positions',  'Iteration t', r'$|x_i(t)|$', r'$|\bar{x} (t)|$', save,  file_name, True)
            else:
                plot_2D_trajectory(self.cloud_history_mean, self.N, save, file_name, limits, self.cloud_history, X, Y, z)
                if log:
                    plot_2D_trajectory(np.absolute(self.cloud_history_mean), self.N, save, file_name, limits, np.absolute(self.cloud_history), X, Y, z, True)

        else:
            if not dimension2:
                cloud_history = [it.flatten() for it in self.cloud_history]
                plot_list(cloud_history, 'Particle positions', 'Iteration t',  r'$x(t)$', save, file_name)
                if log:
                    plot_list(np.absolute(cloud_history), 'Particle positions', 'Iteration t',  r'$|x(t)|$', save, file_name, True)

            else:
                plot_2D_trajectory(self.cloud_history_mean, self.N, save, file_name, limits, None, X, Y, z)
                if log:
                    plot_2D_trajectory(np.absolute(self.cloud_history_mean), self.N, save, file_name, limits, None, X, Y, z, True)


    def plot_everything(self, folder_name, f_dim, log = False, limits =[-5,5,-5,5]):
        
        if not os.path.exists('Images'):
            os.makedirs('Images')
        if not os.path.exists('Images/' + folder_name):
            os.makedirs('Images/' + folder_name)
        
        if f_dim == 1: 
            self.plot_function(save = True, file_name = folder_name + "/function.png", dimension2 = False, limits = limits)
            self.plot_cloud_history(save = True, file_name = folder_name + "/cloud_history.png", dimension2 = False, limits = limits) 
            if log:
                self.plot_cloud_history(save = True, file_name = folder_name + "/cloud_history_log.png", dimension2 = False, limits = limits, X = None, Y = None, z = None, log = True) 

        if "gradient_descent" not in self.train_method and f_dim == 1:
            self.integral_on_gaussian_measure(True, True, "/integral_on_gaussian_measure.png")
            x = np.arange(self.elapsed_iterations)
            data = np.array([self.cloud_history_mean,self.integral_history])
            plot_xy(x, data, '', 'Iteration t', '', [r'$\sum_{i=1}^N f(x_i(t))/{N}$',r'$\int f(x,t) m(x,t) dx$'], False, save = True, file_name = folder_name + "/cost_mean_history2.png")            
            plot_xy(x, data, '', 'Iteration t', '', [r'$\sum_{i=1}^N f(x_i(t))/{N}$',r'$\int f(x,t) m(x,t) dx$'], True, save = True, file_name = folder_name + "/cost_mean_history2_log.png")
        if f_dim == 2:
            X, Y, z  = get_2D_plot_data(self, limits)
            self.plot_function(save = True, file_name = folder_name + "/function.png", dimension2 = True, limits = limits, X = X, Y = Y, z = z)
            self.plot_cloud_history(save = True, file_name = folder_name + "/cloud_history.png", dimension2 = True, limits = limits, X= X, Y = Y, z = z)

        self.plot_cost_history(save = True, file_name = folder_name + "/cost_history.png")
        self.plot_cost_history(save = True, file_name = folder_name + "/cost_history_log.png", log = True)

        if "gradient_descent" not in self.train_method:
            self.plot_cost_mean_history(save = True, file_name = folder_name + "/cost_mean_history.png")
            self.plot_cost_mean_history(save = True, file_name = folder_name + "/cost_mean_history_log.png", log = True)

        if "gradient_descent" not in self.train_method:
            self.plot_var_history(save = True, file_name = folder_name + "/var_history.png")
            self.plot_var_history(save = True, file_name = folder_name + "/var_history_log.png", log = True) 
        
        if "gradient_descent" not in self.train_method:
            self.plot_gradS_history(save = True, file_name = folder_name + "/gradS_history.png")
            self.plot_gradS_history(save = True, file_name = folder_name + "/gradS_history_log.png", log = True)


        self.get_all_parameters(True, "Images/" + folder_name + '/Parameters.csv')



def plot_data(x, fx, title, xlabel, ylabel, legend = None, log = False, save = False, file_name = ""):
    plot_xy(x, fx, title, xlabel, ylabel, legend, log, save, file_name)