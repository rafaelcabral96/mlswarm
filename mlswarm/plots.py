from mlswarm.utils import get_integral_on_gaussian_measure, get_all_parameters_util
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


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
        
    plot_matrix = [[np.linalg.norm(cloudf[i] - cloudf[j], ord=2)() for j in range(N)] for i in range(N)] 
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

def plot_matrix(history, history_mean, title, xlabel, ylabel, legend, save, file_name, log = False, x = None):

    df = pd.DataFrame(history)
    df=df.astype(float)

    plt.figure()
    if log:
        plt.yscale('log')
    for particle in range(len(df.columns)):
        if x is None:
            plt.plot(df.iloc[:,particle], lw = 0.2)
        else:
            plt.plot(x,df.iloc[:,particle], lw = 0.2)

    if x is None:
        plt.plot(history_mean, 'k-',lw= 2)
    else:
        plt.plot(x,history_mean, 'k-',lw= 2)

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



def plot_cost_history_util(self, save = False, file_name = "", log = False):

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

def plot_cost_mean_history_util(self, save = False, file_name = "", log = False):
    if not log:
        plot_list(np.mean(self.cost_history,axis=1), "Average function value", "Iteration t",  r'$\sum_{i=1}^N f(x_i(t))/N$', save, file_name) 
    else:
        plot_list(np.mean(self.cost_history,axis=1), "Average function value", "Iteration t",  r'$|\sum_{i=1}^N f(x_i(t))/N|$', save, file_name, True) 

def plot_var_history_util(self, save = False, file_name = "", log = False):
    if not log:
        plot_list(self.cloud_var_history,'Cloud variance','Iteration t', r'$\sigma (t)$', save, file_name) 
    else:
        plot_list(self.cloud_var_history,'Cloud variance','Iteration t', r'$\sigma (t)$', save, file_name, True) 

def plot_gradS_history_util(self, save = False, file_name = "", log = False):
    if not log:
        plot_list(self.gradS_history, r"$||\nabla S||$" , "Iteration t", r"$||\nabla S|| (t)$", save, file_name)     
    else:
        plot_list(self.gradS_history, r"$||\nabla S||$" , "Iteration t", r"$||\nabla S|| (t)$", save, file_name, True)     


def integral_on_gaussian_measure_util(self, plot = False, save = False, file_name = ""):
    self.integral_history = get_integral_on_gaussian_measure(self)
    if plot:
        plot_list(self.integral_history, 'Function integral on gaussian measure',  'Iteration t',  r'$\int f(x,t) m(x,t) dx$', save,  file_name)

def plot_function_util(self, save = False, file_name = "", dimension2 = False, limits =[-5,5,-5,5], X = None, Y = None, z = None):
    if not dimension2:
        x = np.array([np.linspace(limits[0], limits[1], 2000)])
        fx = self.func(x)
        plot_xy(x.flatten(), [fx.flatten()], 'Function', 'x',  r'$f(x)$',None, False, save, file_name)
    else:
        if X is None:
            X, Y, z  = get_2D_plot_data(self, limits)
        plot_2D_func(X, Y, z, save, file_name)

def plot_cloud_history_util(self, save = False, file_name = "", dimension2 = False, limits = [-5,5,-5,5], X = None, Y = None, z = None, log = False):
        
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

def plot_cost_history2_util(self, save = False, file_name = "", log = False):
        x = np.arange(0,self.function_evaluations, int(self.function_evaluations/self.elapsed_iterations))
        plot_matrix(self.cost_history, self.cost_history_mean, 'Function value',  'Function Evaluations', r'$f(x_i(t))$', r'$f(\bar{x} (t))$', save, log,file_name,x)


def plot_everything_util(self, folder_name, f_dim, log = False, limits =[-5,5,-5,5]):
    
    #check if cloud history information was recorder
    if len(self.cost_history_mean) < 1:
        print("\nPlot Everything Failed - Please set 'track_history' hyperparameter in 'minimize' method to True")
        return

    if not os.path.exists('Images'):
        os.makedirs('Images')
    if not os.path.exists('Images/' + folder_name):
        os.makedirs('Images/' + folder_name)
    
    if f_dim == 1: 
        plot_function_util(self, save = True, file_name = folder_name + "/function.png", dimension2 = False, limits = limits)
        plot_cloud_history_util(self, save = True, file_name = folder_name + "/cloud_history.png", dimension2 = False, limits = limits) 
        if log:
            plot_cloud_history_util(self, save = True, file_name = folder_name + "/cloud_history_log.png", dimension2 = False, limits = limits, X = None, Y = None, z = None, log = True) 

    if "gradient_descent" not in self.train_method and f_dim == 1:
        integral_on_gaussian_measure_util(self, True, True, "/integral_on_gaussian_measure.png")
        x = np.arange(len(self.cloud_history_mean))
        data = np.array([self.cloud_history_mean,self.integral_history])
        plot_xy(x, data, '', 'Iteration t', '', [r'$\sum_{i=1}^N f(x_i(t))/{N}$',r'$\int f(x,t) m(x,t) dx$'], False, save = True, file_name = folder_name + "/cost_mean_history2.png")            
        plot_xy(x, data, '', 'Iteration t', '', [r'$\sum_{i=1}^N f(x_i(t))/{N}$',r'$\int f(x,t) m(x,t) dx$'], True, save = True, file_name = folder_name + "/cost_mean_history2_log.png")
    if f_dim == 2:
        X, Y, z  = get_2D_plot_data(self, limits)
        plot_function_util(self, save = True, file_name = folder_name + "/function.png", dimension2 = True, limits = limits, X = X, Y = Y, z = z)
        plot_cloud_history_util(self, save = True, file_name = folder_name + "/cloud_history.png", dimension2 = True, limits = limits, X= X, Y = Y, z = z)

    plot_cost_history_util(self, save = True, file_name = folder_name + "/cost_history.png")
    if log:
        plot_cost_history_util(self, save = True, file_name = folder_name + "/cost_history_log.png", log = True)

    if "gradient_descent" not in self.train_method:
        plot_cost_mean_history_util(self, save = True, file_name = folder_name + "/cost_mean_history.png")
        if log:
            plot_cost_mean_history_util(self, save = True, file_name = folder_name + "/cost_mean_history_log.png", log = True)

    if "gradient_descent" not in self.train_method:
        plot_var_history_util(self, save = True, file_name = folder_name + "/var_history.png")
        if log:
            plot_var_history_util(self, save = True, file_name = folder_name + "/var_history_log.png", log = True) 
    
    if "gradient_descent" not in self.train_method:
        plot_gradS_history_util(self, save = True, file_name = folder_name + "/gradS_history.png")
        if log:
            plot_gradS_history_util(self, save = True, file_name = folder_name + "/gradS_history_log.png", log = True)

    get_all_parameters_util(self, True, "Images/" + folder_name + '/Parameters.csv')

def plot_data_util(x, fx, title, xlabel, ylabel, legend = None, log = False, save = False, file_name = ""):
    plot_xy(x, fx, title, xlabel, ylabel, legend, log, save, file_name)