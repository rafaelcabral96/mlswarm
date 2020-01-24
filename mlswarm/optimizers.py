import numpy as np
import time
from mlswarm.utils import flatten_weights, flatten_weights_gradients, unflatten_weights, get_var


#ALGORITHMS FOR COMPUTING UPDATE---------------------------------------------

#derivative free version of the swarm algorithm
#particles updated all at once -> consumes more memory, but in most cases is faster
def update_cloud_derivative_free(self, cloudf, cost, lr, N, kernel_a, alpha, beta, gamma):
  
    #compute mean anc variance of cloud
    cloud_mean = np.mean(cloudf, axis=0)
    cloud_var = get_var(cloudf,cloud_mean) 
    cloud_var_mean = np.mean(cloud_var)

    #compute norm, kernel, and gradient of kernel 
    params_diff_matrix = cloudf[:,np.newaxis] - cloudf 
    norm = np.mean(params_diff_matrix**2, axis= 2) 
    kernels = np.exp(-kernel_a*norm/cloud_var_mean)
    gkernels = -2*(kernel_a/cloud_var_mean)*np.einsum('ijk,ij -> ijk',params_diff_matrix,kernels)

    #compute (x - x_mean)/var, a difference variance is considered for each dimension
    omega = np.einsum('ij,j -> ij', cloudf-cloud_mean, 1/cloud_var) 

    gamma1 = gamma
    gamma2 = alpha

    #attraction position for aggregation term -> could be cloud_mean or self.best_particle
    attraction_position = self.best_particle

    update =  ( np.einsum('ij,jk -> ik', kernels, np.einsum('j,jk -> jk',(cost + gamma1),omega) + gamma2*(cloudf-attraction_position))  + np.einsum('j,ijk -> ik', cost + gamma1, gkernels) )  * float(1/N)

    return update, cloud_var


#derivative free version of the swarm algorithm
#particles updated one at the time -> consumes less memory but slower
def update_cloud_derivative_free2(cloudf, cost, lr, N, kernel_a, alpha, beta, gamma):
    gamma1 = gamma
    gamma2 = alpha
    
    cloud_mean = np.mean(cloudf, axis=0)
    cloud_var = get_var(cloudf,cloud_mean) 
    cloud_var_mean = np.mean(cloud_var)
    cost = np.squeeze(np.array(cost))
    omega = np.einsum('ij,j -> ij', cloudf-cloud_mean, 1/cloud_var) #N*N_param,N_param vector

    updates = []
    for i in range(N):
        params_diff = cloudf[i]- cloudf #matrix with NxN_params
        norm = np.sum(params_diff**2, axis= 1) 
        
        kernels = np.exp(-kernel_a*norm/ cloud_var_mean)
        gkernels = -2*(kernel_a/ cloud_var_mean)*np.einsum('jk,j -> jk',params_diff,kernels)
        
        update =  ( np.einsum('j,jk -> k', kernels, np.einsum('j,jk -> jk',(cost + gamma1),omega) + gamma2*(cloudf-cloud_mean) ) +
               np.einsum('j,jk -> k', cost + gamma1, gkernels) ) * float(1/N)
        updates.append(update)

    return np.array(updates), cloud_var


#swarm algorithm using derivatives
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

    return update, cloud_var

#ALGORITHMS FOR COMPUTING NEW CLOUD-----------------------------------------------------------------------

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

def adaptive_step_size(lr,i,update,update_old):
    theta = 0.2
    lr_old = lr
    if i == 0:
        lr = lr_old
    else:
        lr = 2 * lr_old * theta * np.sum(np.abs(update))/np.sum(np.abs((update - update_old)))
        lr = np.min((lr,1))
        lr = np.max((lr,0.00001))
    return lr

