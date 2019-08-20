import numpy as np
import time 
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
#from scipy import stats
import scipy.integrate as integrate
import pandas 


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
        if(np.mean(np.einsum('ij -> i',np.exp(-kernel_a*norm))/N)) < 1:
            break

    print("Kernel constant found: " + str(kernel_a))
    return kernel_a

def gradient(func, x, h = None):
    if h is None:
        # Note the hard coded value found here is the square root of the
        # floating point precision, which can be found from the function
        # call np.sqrt(np.finfo(float).eps).
        h = 1e-08
    xph = x + h
    dx = xph - x
    return (func(xph) - func(x)) / dx

#def gradient_costfunction(cost)

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

def get_var(cloud, cloud_mean = 0):
     return np.var(cloud,axis=0)
     #return np.mean((cloud-cloud_mean)**2)

#def normal_test(cloudf):
#    k, _ = stats.normaltest( (cloudf- np.mean(cloudf,axis=0))/np.var(cloudf,axis=0) , axis = 0)
#    print(np.mean(k))
#    print("Normality test p-value - percentage of particles rejected: "  + str(1 - np.mean(np.greater(p,0.05))))

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


def gaussian(x, mu, var):
    return np.exp(-np.power(x - mu, 2.) / (2 * var))/(np.sqrt(2*np.pi*var))

def integral_on_gaussian_measure_util(self):
    integral_history = []
    for mean, var in zip(self.cloud_history_mean,self.cloud_var_history):
        integrant = lambda x, mean, var: self.func([x]) * gaussian(x,mean,var) 
        integral, _ = integrate.quad(integrant,-np.inf, np.inf, args = (mean[0],var)) #, args = (mean,var))
        integral_history.append(integral)
    return integral_history



#PLOTS-----------------------------------------------

def plot_distance_matrix(cloud, N):
    n_params = len(cloud[0])
    tensor_names = list(cloud[0].keys())

    #flatten all tensors
    cloudf = []
    for nn in range(N):
        flatten = np.ndarray.flatten(cloud[nn][tensor_names[0]])
        for param in range(n_params):
            flatten_temp = np.ndarray.flatten(cloud[nn][tensor_names[param]])
            flatten = np.concatenate((flatten,flatten_temp),axis=None)
        cloudf.append(flatten)    
        
    plot_matrix = [[norm2(cloudf[i],cloudf[j]) for j in range(N)] for i in range(N)] 
    plt.imshow(np.asarray(plot_matrix))
    plt.title("Distance between NN")
    plt.colorbar()

    plt.show()
    plt.close()

def plot_list(history, title, xlabel, ylabel, save, file_name, log = False):
    plt.figure()
    plt.plot(history)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if log:
        plt.yscale('log')
    plt.title(title)

    if save:
        if not os.path.exists('Images'):
            os.makedirs('Images')
        plt.savefig('Images/' + file_name, dpi = 300)

    plt.show()
    plt.close()

def plot_matrix(history, history_mean, title, xlabel, ylabel, legend, save, file_name, log = False):

    df = pandas.DataFrame(history)
    df=df.astype(float)

    plt.figure()
    if log:
        plt.yscale('log')
    for particle in range(len(df.columns)):
        plt.plot(df.iloc[:,particle], lw = 0.2)

    plt.plot(history_mean, 'k-',lw= 2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.legend([Line2D([0], [0], color = 'black', lw= 2)],[legend])

    if save:
        if not os.path.exists('Images'):
            os.makedirs('Images')
        plt.savefig('Images/' + file_name, dpi = 300)

    plt.show()
    plt.close()

def plot_2D_trajectory(cloud_history_mean, N, save, file_name, limits, cloud_history = None, X = None, Y = None, z = None, log = False):


    plt.figure()
    plt.contourf(X, Y, z, cmap=plt.get_cmap("bwr"))

    if cloud_history is not None:
        for i in range(N):
            cloud_history_x = [cloud_history[itera][0][i] for itera in range(len(cloud_history))]
            cloud_history_y = [cloud_history[itera][1][i] for itera in range(len(cloud_history))]
            plt.plot(cloud_history_x, cloud_history_y, '-', lw = 0.3)
            plt.plot(cloud_history_x[0], cloud_history_y[0], 'bo', markersize = 0.5) 
    
    plt.xlim(limits[0],limits[1])
    plt.ylim(limits[2],limits[3])
    plt.xlabel("x")
    plt.ylabel("y")
    if log:
        plt.xscale('log')
        plt.yscale('log')
    plt.legend([Line2D([0], [0], marker = 'o', markersize=10, color = 'black')],['Cloud mean'])
    plt.axvline(x=0, color='k', linestyle='--', lw = 1)
    plt.axhline(y=0, color='k', linestyle='--', lw = 1)
    plt.title(r'Particle positions  $ x_i (t)$')
    
    cloud_history_mean_x = [it[0] for it in cloud_history_mean]                
    cloud_history_mean_y = [it[1] for it in cloud_history_mean]   
    plt.plot(cloud_history_mean_x, cloud_history_mean_y, 'k-', lw = 2)
    plt.plot(cloud_history_mean_x[0], cloud_history_mean_y[0], 'ko')
    plt.colorbar()

    if save:
        if not os.path.exists('Images'):
            os.makedirs('Images')
        plt.savefig('Images/' + file_name, dpi = 300)

    plt.show()
    plt.close()

def plot_xy(x, fx, title, xlabel, ylabel, legend, log, save, file_name):
    fig = plt.figure()
    if log:
        plt.yscale('log')
    ax = plt.subplot(111)
    for f in fx:   
        ax.plot(x,f)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if legend is not None: 
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(legend, loc='center left', bbox_to_anchor=(1, 0.5))

    if save:
        if not os.path.exists('Images'):
            os.makedirs('Images')
        plt.savefig('Images/' + file_name, dpi = 300)

    plt.show()
    plt.close()

def get_2D_plot_data(self,limits):
    x_array = np.array(np.linspace(limits[0], limits[1], 2000))
    y_array = np.array(np.linspace(limits[0], limits[1], 2000))
    z = np.zeros((2000,2000))
    for i in range(len(x_array)):
        for j in range(len(y_array)):
            z[i,j] = self.func(np.array([x_array[i],y_array[j]]))
    X, Y = np.meshgrid(x_array, y_array)

    plt.show()
    plt.close()

    return X, Y, z

def plot_2D_func(X, Y, z, save, file_name):
    from mpl_toolkits.mplot3d import axes3d 

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, z, cmap=plt.get_cmap("bwr"))

    if save:
        if not os.path.exists('Images'):
            os.makedirs('Images')
        fig.savefig('Images/' + file_name, dpi = 300)
    
    plt.show(fig)
    plt.close(fig)

 