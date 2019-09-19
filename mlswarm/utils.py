import numpy as np
import scipy.integrate as integrate
import time 
#from scipy import stats


#UTIL FUNCTIONS FOR BOTH CLASSES------------------------------------------------------

def timerfunc(func):
    #timer decorator
    def function_timer(*args, **kwargs):
        start = time.clock()
        start_cputime = time.process_time()

        value = func(*args, **kwargs)

        end_cputime = time.process_time()
        end = time.clock() 
        runtime_cpu = end_cputime - start_cputime
        runtime = end - start  
        msg = "\nThe clock time (CPU time) for {func} was {time:.5f} ({cpu_time:.5f}) seconds"
        print(msg.format(func=func.__name__,
                         time=runtime,
                         cpu_time = runtime_cpu))
        
        args[0].cpu_time = runtime_cpu
        args[0].clock_time = runtime

        return value
    return function_timer

def get_var(cloud, cloud_mean = 0):
     return np.sum( (cloud - cloud_mean)**2, axis=0)/cloud.shape[0] #use biased version


def gradient(func, x, h = None):
    if h is None:
        h = 1e-08
    xph = x + h
    dx = xph - x
    return (func(xph) - func(x)) / dx


def gradS(cloud,kernel_a):
    cloud_mean = np.mean(cloud, axis=0)
    cloud_var = get_var(cloud) #np.var(cloudf, axis=0) works best

    params_diff_matrix = cloud[:,np.newaxis] - cloud    
    #compute kernels
    norm = np.sum(params_diff_matrix**2, axis= 2) 
    kernels = np.exp(-kernel_a*norm)
    gkernels = -2*kernel_a*np.einsum('ijk,ij -> ijk',params_diff_matrix,kernels)

    gradS = np.einsum('ij,jk -> ik', kernels, (cloud - cloud_mean)/cloud_var) + np.einsum('ijk ->ik',gkernels)
    gradS = np.sqrt(np.sum(gradS**2))

    return gradS

##UTILS FUNCTIONS FOR NEURALNET CLASS---------------------------------------------------------

def flatten_weights(cloud,N):
    tensor_names = list(cloud[0].keys())

    nn_shape = []
    for param in cloud[0]:
        nn_shape.append(np.shape(cloud[0][param]))

    cloudf = []
    for nn in range(N):
        flatten = np.array([])
        for param in cloud[nn].values():
            flatten = np.concatenate((flatten,np.ndarray.flatten(param)),axis=None)
        cloudf.append(flatten)   
    cloudf = np.array(cloudf)

    return cloudf, nn_shape, tensor_names

def flatten_weights_gradients(cloud, gradients, N):
    n_params = len(cloud[0])
    weight_names = list(cloud[0].keys())
    nn_shape = []
    for param in cloud[0]:
        nn_shape.append(np.shape(cloud[0][param]))
            
    cloudf = []
    gradientsf = []
    for nn in range(N):
        flatten = np.array([])
        flatten_g = np.array([])
        for param in range(n_params):
            flatten_temp = np.ndarray.flatten(cloud[nn][weight_names[param]])
            flatten = np.concatenate((flatten,flatten_temp),axis=None)
            flatten_g_temp =  np.ndarray.flatten(gradients[nn]["d" + weight_names[param]])
            flatten_g = np.concatenate((flatten_g,flatten_g_temp),axis=None)
        cloudf.append(flatten)    
        gradientsf.append(flatten_g)
    cloudf = np.array(cloudf)
    gradientsf = np.array(gradientsf)

    return cloudf, gradientsf, nn_shape, weight_names

def unflatten_weights(cloudf,shapes,weight_names,N):
    n_params = len(shapes)
    new_nn_weights = []
    for nn in range(N):
        init = 0
        new_params_nn = {}
        for param in range(n_params):
            num_params = int(np.prod(shapes[param]))
            new_params_nn[weight_names[param]] = np.reshape(cloudf[nn][init:(init + num_params)],shapes[param])
            init += num_params
        new_nn_weights.append(new_params_nn)

    return new_nn_weights

def kernel_a_finder(cloudf, N):

    print("Finding kernel constant...")

    cloud_diff_matrix = cloudf[:,np.newaxis] - cloudf
    norm = np.sum(cloud_diff_matrix**2, axis=2) 
    kernel_a_pool = [0.00001, 0.0005, 0.0001, 0.0005,0.001,0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50, 100]
    for kernel_a in kernel_a_pool:
        if(np.mean(np.einsum('ij -> i',np.exp(-kernel_a*norm))/N)) < 0.75:
            break

    print("Kernel constant found: " + str(kernel_a))
    return kernel_a

def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_

def get_mean(cloud):
    params_mean = {}
    for key in cloud[0]:
        params_mean[key] = np.mean([cloud[j][key] for j in range(len(cloud))],axis=0)
    return params_mean



#UTIL FUNCTIONS FOR FUNCTION CLASS -------------------------------------------------------------

def get_parameters_nn(self, X, Y, d):

    method = d.get('method', 'gradient_free')
    algorithm = d.get('algorithm', 'euler')
    max_epochs =  d.get('max_epochs', 500)
    n_batches = d.get('n_batches', 1)
    batch_size = d.get('batch_size', X.shape[0])
    learning_rate = d.get('learning_rate', 0.01)
    cost_type = d.get('cost_type','rmse')
    kernel_a = d.get('kernel_a', 1)
    alpha_init = d.get('alpha_init', 0)
    alpha_rate = d.get('alpha_rate', 1)
    beta = d.get('beta', 0)
    gamma = d.get('gamma', 0)
    verbose = d.get('verbose', False)
    var_epsilon = d.get('var_epsilon',0)

    self.train_method = method
    self.algorithm = algorithm

    #transposing input and output data
    X = X.T
    Y = Y.T

    return [self, X, Y, method, max_epochs, n_batches, batch_size, learning_rate, cost_type, kernel_a,  alpha_init, alpha_rate, beta, gamma, verbose, var_epsilon]

def get_parameters(self, d):
    max_iterations =  d.get('max_iterations', 10000)
    epsilon = d.get('epsilon', 0) 
    var_epsilon = d.get('var_epsilon',0.0001)
    learning_rate = d.get('learning_rate', 0.01)
    method = d.get('method', 'gradient_free')
    algorithm = d.get('algorithm', 'euler_adaptive')
    restart_cloud = d.get('restart_cloud', False)
    kernel_a = d.get('kernel_a', 1)
    alpha_init = d.get('alpha_init', 0)
    alpha_rate = d.get('alpha_rate', 1)
    beta = d.get('beta', 0)
    gamma_init = d.get('gamma_init', 0)
    gamma_rate = d.get('gamma_init', 0)
    verbose = d.get('verbose', False)
    track_history = d.get('track_history', True)

    self.train_method = method
    self.algorithm = algorithm
    self.train_parameters = [max_iterations, epsilon, var_epsilon, learning_rate, kernel_a, alpha_init, alpha_rate, beta, gamma_init, gamma_rate]

    return [self, max_iterations, epsilon, var_epsilon, learning_rate, method, restart_cloud, kernel_a, alpha_init, alpha_rate, beta, gamma_init, gamma_rate, verbose, track_history]

def gaussian(x, mu, var):
    return np.exp(-np.power(x - mu, 2.) / (2 * var))/(np.sqrt(2*np.pi*var))

def get_integral_on_gaussian_measure(self):
    integral_history = []
    for mean, var in zip(self.cloud_history_mean,self.cloud_var_history):
        integrant = lambda x, mean, var: self.func([x]) * gaussian(x,mean,var) 
        integral, _ = integrate.quad(integrant,-np.inf, np.inf, args = (mean[0],var)) #, args = (mean,var))
        integral_history.append(integral)
    return integral_history


def get_all_parameters_util(self,save = False, file_name = ""):

    header = ["name", "N", "cloud_var", "cloud_mean" , "cloud_mean_func" ,"train_method", "train_algorithm", "elapsed_iterations", "function_evaluations", "clock_time", "cpu_time", "max_iterations", "epsilon", "var_epsilon", "learning_rate", "kernel_a", "alpha_init", "alpha_rate", "beta", "gamma", "gamma_rate"]
    data = [self.name, self.N, self.cloud_var,np.array2string(self.cloud_mean, precision = 5), np.format_float_scientific(self.cloud_mean_func.flatten()) ,self.train_method, self.algorithm, self.elapsed_iterations, self.function_evaluations, self.clock_time, self.cpu_time]
    data = data + self.train_parameters

    if save:
        import pandas 
        df = pandas.DataFrame([data], columns = header)
        df.to_csv(file_name, sep=',', index=False)
    else:
        return header, data