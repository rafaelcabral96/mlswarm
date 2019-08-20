import numpy as np
import time
from mlswarm.utils import flatten_weights, flatten_weights_gradients, unflatten_weights, get_var



def update_cloud_derivative_free(cloudf, cost, lr, N, kernel_a, alpha, beta, gamma):
  
    #compute mean and standart deviation and difference matrix between particles
    cloud_mean = np.mean(cloudf, axis=0)
    cloud_var = get_var(cloudf, cloud_mean) #np.var(cloudf, axis=0) works best

    #fast but memory monster: If N=n_variables=8000 a 8000x8000x8000 tensor will be initialized = 13 TB of RAM
   
    params_diff_matrix = cloudf[:,np.newaxis] - cloudf 

    #compute kernels
    norm = np.mean(params_diff_matrix**2, axis= 2) 
    kernels = np.exp(-kernel_a*norm/(np.mean(cloud_var) + 0.0001))
    #norm = np.sum(params_diff_matrix**2, axis= 2) 
    #kernels = np.exp(-kernel_a*norm)
    gkernels = -2*kernel_a*np.einsum('ijk,ij -> ijk',params_diff_matrix,kernels)

    cost = np.squeeze(np.array(cost))
    omega = np.divide(cloudf-cloud_mean,cloud_var) #N*N_param,N_param vector

    gamma1 = gamma
    gamma2 = alpha

    print(cloud_mean)

    update =  ( np.einsum('ij,jk -> ik', kernels, np.einsum('j,jk -> jk',(cost + gamma1),omega) + gamma2*(cloudf-cloud_mean) ) +
           np.einsum('j,ijk -> ik', cost + gamma1, gkernels) ) * float(1/N)

    return update

def update_cloud_derivative_free3(cloudf, cost, lr, N, kernel_a, alpha, beta, gamma):
  
    #compute mean and standart deviation and difference matrix between particles
    cloud_mean = np.mean(cloudf, axis=0)
    cloud_var = get_var(cloudf) #np.var(cloudf, axis=0) works best

    #fast but memory monster: If N=n_variables=8000 a 8000x8000x8000 tensor will be initialized = 13 TB of RAM
   
    params_diff_matrix = cloudf[:,np.newaxis] - cloudf 

    N_sample = 10
    N_params = cloudf.shape[1]

    params_diff_matrix = np.empty(shape = (0,N_sample,N_params), dtype = float)
    cost_matrix = np.empty(shape = (0,N_sample), dtype = float)
    omega_matrix = np.empty(shape = (0,N_sample,N_params), dtype = float)
    cloudf_sample = np.empty(shape = (0,N_sample,N_params), dtype = float)
    for i in range(N):
        sample = np.random.choice(N,N_sample)
        params_diff_matrix_i = cloudf[i] - cloudf[sample]
        cost_i = cost[sample]
        omega_i = np.divide(cloudf[sample]-cloud_mean,cloud_var)
        cloudf_i = cloudf[sample] - cloud_mean

        params_diff_matrix = np.append(params_diff_matrix, [params_diff_matrix_i], axis=0)
        cost_matrix = np.append(cost_matrix, [cost_i], axis=0)
        omega_matrix = np.append(omega_matrix, [omega_i], axis=0)
        cloudf_sample = np.append(cloudf_sample, [cloudf_i], axis=0)

    print(params_diff_matrix.shape)
    print(cost_matrix.shape)
    print(omega_matrix.shape)
    print(cloudf_sample.shape)

    #compute kernels
    norm = np.sum(params_diff_matrix**2, axis= 2) 
    kernels = np.exp(-kernel_a*norm)
    gkernels = -2*kernel_a*np.einsum('ijk,ij -> ijk',params_diff_matrix,kernels)

    gamma1 = gamma
    gamma2 = alpha

    update =  ( np.einsum('ij,ijk -> ik', kernels, np.einsum('ij,ijk -> ijk',(cost_matrix + gamma1),omega_matrix) + gamma2*(cloudf_sample-cloud_mean) ) +
           np.einsum('ij,ijk -> ik', cost_matrix + gamma1, gkernels) ) * float(1/N)

    return update


def update_cloud_derivative_free2(cloudf, cost, lr, N, kernel_a, alpha, beta, gamma):
    #alternative: only initialize a 2000x8000 tensor at a time = 1.2 GB of Ram
    gamma1 = gamma
    gamma2 = alpha
    
    cloud_mean = np.mean(cloudf, axis=0)
    cloud_var = get_var(cloudf)
    omega = np.divide(cloudf-cloud_mean,cloud_var) #N*N_param,N_param vector
    cost = np.squeeze(np.array(cost)) # N vector

    updates = []
    for i in range(N):
        params_diff = cloudf[i]- cloudf #matrix with NxN_params
        norm = np.sum(params_diff**2, axis= 1) 
        kernels = np.exp(-kernel_a*norm) #vector with N elements -> kernel between particle i (fixed) and j
        gkernels = -2*kernel_a*np.einsum('jk,j -> jk',params_diff,kernels)
        update =  ( np.einsum('j,jk -> k', kernels, np.einsum('j,jk -> jk',(cost + gamma1),omega) + gamma2*(cloudf-cloud_mean) ) +
               np.einsum('j,jk -> k', cost + gamma1, gkernels) ) * float(1/N)
        updates.append(update)

    return np.array(updates)


def update_sub_cloud_derivative_free(cloudf, cost, lr, N, kernel_a, alpha, beta, gamma, N_sample):
    #alternative: only initialize a 2000x8000 tensor at a time = 1.2 GB of Ram
    gamma1 = gamma
    gamma2 = alpha
    
    cloud_mean = np.mean(cloudf, axis=0)
    cloud_var = get_var(cloudf)
    omega = np.divide(cloudf - cloud_mean, cloud_var)
    cost = np.squeeze(np.array(cost))

    updates = []

    for i in range(N):
    
        sample = np.random.randint(0,N,N_sample)
        cloudf_sample = np.take(cloudf,sample,axis=0)
        cost_sample = np.take(cost,sample,axis=0) # N vector
        omega_sample = np.take(omega,sample,axis=0)

        params_diff = np.take(cloudf,i,axis=0) - cloudf_sample #matrix with NxN_params

        #print(cloudf_sample.shape)
        #print(params_diff.shape)
        norm = np.sum(params_diff**2, axis= 1) 
        kernels = np.exp(-kernel_a*norm) #vector with N elements -> kernel between particle i (fixed) and j
        gkernels = -2*kernel_a*np.einsum('jk,j -> jk',params_diff,kernels)
        update =  ( np.einsum('j,jk -> k', kernels, np.einsum('j,jk -> jk',(cost_sample + gamma1),omega_sample) + gamma2*(cloudf_sample-cloud_mean) ) +
               np.einsum('j,jk -> k', cost_sample + gamma1, gkernels) ) * float(1/N)
        updates.append(update)

    return np.array(updates)
def update_cloud(cloudf, gradientsf, lr, N, kernel_a, alpha, beta, gamma):

    #compute mean and standart deviation and difference matrix between particles
    cloud_mean = np.mean(cloudf, axis=0)
    cloud_var = get_var(cloudf) 
    params_diff_matrix = cloudf[:,np.newaxis] - cloudf
    
    #compute kernels
    norm = np.sum(params_diff_matrix**2, axis=2) #no sqrt
    kernels = np.exp(-kernel_a*norm)
    gkernels = -2*kernel_a*np.einsum('ijk,ij -> ijk',params_diff_matrix,kernels)

    Q = np.einsum('ij,jk -> ik', kernels, gradientsf) * float(1/N) 

    if alpha > 0 :
        R = np.einsum('ij,jlk -> ik',kernels,params_diff_matrix) * float(1/N**2)
    else:
        R = 0

    if beta > 0 :
        P = np.einsum('ijk -> ik',gkernels) * float(1/N)
    else:
        P = 0

    if gamma > 0 :
        S = np.einsum('ij,jk -> ik', kernels, np.divide(cloudf-cloud_mean,cloud_var)) * float(1/N)
    else:
        S = 0

    update = Q + alpha*R + beta*P + gamma*S

    return update

def update_cloud2(cloudf, gradientsf, lr, N, kernel_a, alpha, beta, gamma):

    #no cant do because of alpha term you need comple NxNxN_param matrix

    return update, cloud_var

def nesterov(cloudf, yf, lamb_prev, elapsed_iterations, lr, update):
    
    if elapsed_iterations == 0:
      lamb = 0
      yf = cloudf
    else:
      lamb = (1.0 + np.sqrt(1.0 + 4.0*lamb_prev**2))/2.0
    lamb_next = (1.0 + np.sqrt(1.0 + 4.0*lamb**2))/2.0

    gamma = (1.0 - lamb) / lamb_next
    yfnext = cloudf - lr * update
    cloudf = (1.0 - gamma)*yfnext + gamma*yf

    return cloudf, yfnext, lamb

# Optimizers for neuralnet object-----------------------------------------------------------------

#euler method--------------------
def update_nn_weights(cloud, gradients, lr, N,  kernel_a, alpha, beta, gamma):
    
    #get cloud (flattened weights), gradients, nn shape and weight names
    cloudf, gradientsf, nn_shape, weight_names = flatten_weights_gradients(cloud, gradients, N)
    
    #get update value to be applied on cloud
    update = update_cloud(cloudf, gradientsf, lr, N, kernel_a, alpha, beta, gamma) 

    #use euler method 
    cloudf -= lr * update

    #restore NN weight shapes 
    new_nn_weights = unflatten_weights(cloudf,nn_shape,weight_names,N)
    
    return new_nn_weights, cloud_var

def update_nn_weights_derivative_free(cloud, cost, lr, N, kernel_a, alpha, beta, gamma): 

    #get cloud (flattened weights), gradients, nn shape and weight names
    cloudf, nn_shape, weight_names = flatten_weights(cloud,N)

    #get update value to be applied on cloud
    update = update_cloud_derivative_free2(cloudf, cost, lr, N, kernel_a, alpha, beta, gamma)

    #use euler method 
    cloudf -= lr * update

    #restore NN weight shapes 
    new_nn_weights = unflatten_weights(cloudf,nn_shape,weight_names,N)
    
    return new_nn_weights

def update_nn_gd(params_values, grads_values, nn_architecture, lr):
    
    # iteration over network layers
    for layer_idx, layer in enumerate(nn_architecture, 1):
        params_values["W" + str(layer_idx)] -= lr * grads_values["dW" + str(layer_idx)]        
        params_values["b" + str(layer_idx)] -= lr * grads_values["db" + str(layer_idx)]

    return [params_values], np.nan

#nesterov method-----------
def update_nn_weights_nesterov(cloud, yf, lamb_prev, elapsed_iterations, gradients, lr, N, kernel_a, alpha, beta, gamma):
   
    #get cloud (flattened weights), gradients, nn shape and weight names
    cloudf, gradientsf, nn_shape, weight_names = flatten_weights_gradients(cloud, gradients, N)
    
    #get update value to be applied on cloud
    update = update_cloud(cloudf, gradientsf, lr, N, kernel_a, alpha, beta, gamma) 

    #use nesterov method
    cloudf, yfnext, lamb = nesterov(cloudf, yf, lamb_prev, elapsed_iterations, lr, update)

    #restore NN weight shapes 
    new_nn_weights = unflatten_weights(cloudf, nn_shape, weight_names, N)
    
    return new_nn_weights, yfnext, lamb


def update_nn_weights_derivative_free_nesterov(cloud, yf, lamb_prev, elapsed_iterations, cost, lr, N, kernel_a, alpha, beta, gamma):
   
    #get cloud (flattened weights), gradients, nn shape and weight names
    cloudf, nn_shape, weight_names = flatten_weights(cloud,N)

    #get update value to be applied on cloud
    update = update_cloud_derivative_free2(cloudf, cost, lr, N, kernel_a, alpha, beta, gamma)
   
    #use nesterov method
    cloudf, yfnext, lamb = nesterov(cloudf, yf, lamb_prev, elapsed_iterations, lr, update)
   
    #restore NN weight shapes 
    new_nn_weights = unflatten_weights(cloudf,nn_shape,weight_names,N)
    
    return new_nn_weights, yfnext, lamb


def update_nn_gd_nesterov(params_values, yf, lamb_prev, elapsed_iterations, grads_values, nn_architecture, lr):

    #get cloud (flattened weights), gradients, nn shape and weight names
    cloudf, gradientsf, nn_shape, weight_names = flatten_weights_gradients(cloud, gradients, N)
    
    #use nesterov method
    cloudf, yfnext, lamb = nesterov(cloudf, yf, lamb_prev, elapsed_iterations, lr, gradientsf)
   
    #restore NN weight shapes 
    new_nn_weights = unflatten_weights(cloudf, nn_shape, weight_names, N)
    
    return new_nn_weights, yfnext, lamb



# Optimizers for Function object-----------------------------------------------------------------

#euler method--------------------
def update_swarm_func(cloud, gradients, lr, N, kernel_a, alpha, beta, gamma):
    update  = update_cloud(cloud, gradients, lr, N, kernel_a, alpha, beta, gamma) 
    cloud -= lr * update
    return cloud, update

def update_swarm_derivative_free_func(cloud, cost, lr, N, kernel_a, alpha, beta, gamma, N_sample = False):
    if N_sample is not False:
        update = update_sub_cloud_derivative_free(cloud, cost, lr, N, kernel_a, alpha, beta, gamma, N_sample)
    else:
        update = update_cloud_derivative_free(cloud, cost, lr, N, kernel_a, alpha, beta, gamma) 

    cloud -= lr * update
    return cloud, update

def update_gd_func(cloud, gradients, lr):
    cloud -= lr * gradients
    return cloud, gradients

def update_swarm_i(cloudf, i, gradients, lr, N, kernel_a, alpha, beta, gamma):
    gamma1 = gamma
    gamma2 = alpha
    
    cloud_mean = np.mean(cloudf, axis=0)
    cloud_var = get_var(cloudf) 
    omega = np.divide(cloudf-cloud_mean,cloud_var) #N*N_param,N_param vector
    cost = np.squeeze(np.array(cost)) # N vector

    updates = []
    params_diff= cloudf[i]-cloudf #matrix with NxN_params
    norm = np.sum(params_diff**2, axis= 1) 
    kernels = np.exp(-kernel_a*norm) #vector with N elements -> kernel between particle i (fixed) and j
    gkernels = -2*kernel_a*np.einsum('jk,j -> jk',params_diff,kernels)
    update =  ( np.einsum('j,jk -> k', kernels, np.einsum('j,jk -> jk',(cost + gamma1),omega) + gamma2*(cloudf-cloud_mean) ) +
           np.einsum('j,jk -> k', cost + gamma1, gkernels) ) * float(1/N)

    return update


#nesterov method--------------------
def update_swarm_func_nesterov(cloud, yf, lamb_prev, elapsed_iterations, gradients, lr, N, kernel_a, alpha, beta, gamma):
    update  = update_cloud(cloud, gradients, lr, N, kernel_a, alpha, beta, gamma) 
    cloud, yfnext, lamb = nesterov(cloud, yf, lamb_prev, elapsed_iterations, lr, update)
    return cloud, yfnext, lamb

def update_swarm_derivative_free_func_nesterov(cloud, yf, lamb_prev, elapsed_iterations, cost, lr, N, kernel_a, alpha, beta, gamma):
    update  = update_cloud_derivative(cloud, cost, lr, N, kernel_a, alpha, beta, gamma)
    cloud, yfnext, lamb = nesterov(cloud, yf, lamb_prev, elapsed_iterations, lr, update)
    return cloud, yfnext, lamb

def update_gd_func_nesterov(cloud, yf, lamb_prev, elapsed_iterations, gradients, lr):
    cloud, yfnext, lamb = nesterov(cloud, yf, lamb_prev, elapsed_iterations, lr, gradients)
    return cloud, yfnext, lamb
