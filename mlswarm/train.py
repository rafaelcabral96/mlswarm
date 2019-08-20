import numpy as np
from mlswarm.neural_networks import init_layers, full_forward_propagation, full_backward_propagation, get_cost_func, get_cost_func_deriv, get_accuracy_value
from mlswarm.utils import timerfunc, get_mean, get_var
from mlswarm.optimizers import update_nn_gd, update_nn_weights, update_nn_weights_derivative_free, update_nn_gd_nesterov, update_nn_weights_nesterov, update_nn_weights_derivative_free_nesterov



@timerfunc
def train_nn(self, X, Y, cloud, nn_architecture, method, max_epochs, n_batches,  batch_size, learning_rate, 
          cost_type, N, kernel_a, alpha_init, alpha_rate, beta, gamma, verbose, var_epsilon):
    
    #check if backprogation is required
    gradient_required = (method in ["gradient_descent","gradient_descent_nesterov", "swarm", "swarm_nesterov"])

    #define necessary history variables for nesterov method
    if "nesterov" in method:
        lamb = 0
        yf = 0

    # initiation of lists storing the cost history 
    cost_history = []
    cost_history_mean = []
    
    alpha = alpha_init
    elapsed_iterations = 0   

    #get cost function 
    cost_func = get_cost_func(cost_type)

    #get derivative of of cost function
    if gradient_required:
        cost_func_deriv = get_cost_func_deriv(cost_type) 

    print("\nTraining started...")   

    for i in range(max_epochs):
        
        for batch in range(n_batches):
   
            start = batch*batch_size
            end = start + batch_size

            Y_hat = []
            costs = []
            cache = []
            grads = []

            #cycle all particles
            for j in range(N):  

                # step forward
                Y_hat_temp, cache_temp = full_forward_propagation(X[:,start:end], cloud[j], nn_architecture)
                Y_hat.append(Y_hat_temp)
                cache.append(cache_temp)
                
                # calculating cost and saving it to history
                costj = cost_func(Y_hat[j], Y[:,start:end]) 
                costs.append(costj)      

                # step backward - calculating gradient
                if gradient_required:
                    gradsj =  full_backward_propagation(Y_hat[j], Y[:,start:end], cache[j], cloud[j], nn_architecture, cost_func_deriv)
                    grads.append(gradsj)  
           
            if method == "swarm":                        cloud, cloud_var = update_nn_weights(cloud, grads, learning_rate, N, kernel_a, alpha, beta, gamma)
            elif method == "swarm_derivfree":            cloud, cloud_var = update_nn_weights_derivative_free(cloud, costs, learning_rate, N, kernel_a, alpha, beta, gamma)
            elif method == "gradient_descent":           cloud, cloud_var = update_nn_gd(cloud[0], grads[0], nn_architecture, learning_rate)
            elif method == "swarm_nesterov":             cloud, yf, lamb, cloud_var = update_nn_weights_nesterov(cloud, yf, lamb, elapsed_iterations, grads, learning_rate, N, kernel_a, alpha, beta, gamma)
            elif method == "swarm_derivfree_nesterov":   cloud, yf, lamb, cloud_var = update_nn_weights_derivative_free_nesterov(cloud, yf, lamb, elapsed_iterations, costs, learning_rate, N, kernel_a, alpha, beta, gamma)
            #elif method == "gradient_descent_nesterov":  cloud, yf, lamb, cloud_var = update_nn_gd_nesterov(cloud[0], yf, lamb, elapsed_iterations, grads[0], nn_architecture, learning_rate)
            else: raise Exception("No method found")

            #end of iteration       
            cost_history.append(costs)

            #mean particle position and its cost
            cloud_mean = get_mean(cloud)
            Y_hat_mean, _ = full_forward_propagation(X[:,start:end], cloud_mean, nn_architecture)
            cost_mean = cost_func(Y_hat_mean, Y[:,start:end])
            cost_history_mean.append(cost_mean)

            elapsed_iterations += 1


        #end of epoch----------------
        cloud_var = np.mean(get_var(cloud_var)) #mean of variances along dimensions of parameter space

        if(verbose):
            print("Iteration: {:05} - Cloud mean cost: {:.5f} - Cloud variance: {:.5f}".format(i, cost_mean, cloud_var))

        alpha += alpha_rate

        if cloud_var < var_epsilon: 
            print("Convergence achieved - particles are localized")
            break

    if i == (max_epochs - 1): print("Maximum amount of epochs reached")

    print("\nCost function value at cloud mean: " + str(cost_mean))
    print("Cost function value (derivative) evaluated {:01} ({:01}) times".format(int(elapsed_iterations*N),int(gradient_required)*elapsed_iterations*N))

    return cloud, cloud_mean, cloud_var, cost_history, cost_history_mean



@timerfunc
def train_fmin2(self, max_iterations, var_epsilon, learning_rate, method, N_sample, kernel_a, alpha_init, alpha_rate, beta, gamma_init, gamma_rate, verbose, track_history):

    from mlswarm.optimizers import update_swarm_func, update_swarm_derivative_free_func, update_gd_func, update_swarm_func_nesterov, update_swarm_derivative_free_func_nesterov, update_gd_func_nesterov
    from mlswarm.utils import gradient, gradS
  
    #cheching if it is necessary to compute gradients
    gradient_required = (method in ["gradient_descent","gradient_descent_nesterov", "swarm", "swarm_nesterov"]) #or time_step_control

    #define necessary history variables for nesterov method
    if "nesterov" in method:
        lamb = 0
        yf = 0

    #init of history lists
    if track_history:
        self.cost_history = []
        self.cost_history_mean = []
        self.cloud_history = []
        self.cloud_history_mean = []
        self.cloud_var_history = []
        self.gradS_history = []

    #compute necessary values
    alpha = alpha_init
    gamma = gamma_init
    self.elapsed_iterations = 0    
    self.function_values = self.func(self.cloud.T)

    updates = []
    for i in range(max_iterations):
        
        #compute gradients
        if gradient_required:
            gradients = gradient(self.func, self.cloud.T).T

        #update cloud
        #for learning_rate in [1,0.1,0.01,0.001]:        
        if method == "swarm":                       cloud_temp, update = update_swarm_func(self.cloud + 0.0, gradients, learning_rate, self.N, kernel_a, alpha, beta, gamma)
        elif method == "swarm_derivfree":           cloud_temp, update = update_swarm_derivative_free_func(self.cloud + 0.0, self.function_values + 0.0, learning_rate, self.N, kernel_a, alpha, beta, gamma, N_sample)
        elif method == "gradient_descent":          cloud_temp, update = update_gd_func(self.cloud + 0.0, gradients, learning_rate)
            #elif method == "swarm_nesterov":            cloud, yf, lamb, cloud_var = update_swarm_func_nesterov(cloud, yf, lamb, elapsed_iterations, gradients, learning_rate, N, kernel_a, alpha, beta, gamma)
            #elif method == "swarm_derivfree_nesterov":  cloud, yf, lamb, cloud_var = update_swarm_derivative_free_func_nesterov(cloud, yf, lamb, elapsed_iterations, function_values, learning_rate, N, kernel_a, alpha, beta, gamma)
            #elif method == "gradient_descent_nesterov": cloud, yf, lamb, cloud_var = update_gd_func_nesterov(cloud, yf, lamb, elapsed_iterations, gradients, learning_rate)
        else: raise Exception("No method found")
            
         #   if (np.mean(self.func(cloud_temp.T)) - np.mean(self.function_values)) < (-learning_rate* np.mean(np.power(np.abs(update),1)) / 2.0 ):
         #       self.cloud = cloud_temp
         #       break           

        updates.append(update) # t x N x N_variables
        self.cloud = cloud_temp - np.divide(learning_rate,1 + np.sqrt(np.sum(np.array(updates)**2, axis=0)))*update
        #print(np.divide(1,np.sqrt(np.sum(np.array(updates)**2, axis=0))))
        #if learning_rate == 0.001:
        #    self.cloud = cloud_temp

        #compute cloud properties
        self.function_values = self.func(self.cloud.T)
        self.cloud_var = np.mean(get_var(self.cloud))
        self.cloud_mean = np.mean(self.cloud, axis=0)
        self.cloud_mean_func = self.func(self.cloud_mean)

        #update alpha and elapsed_iterations
        alpha = alpha + alpha_rate
        gamma = gamma + gamma_rate
        self.elapsed_iterations += 1

        #appending results to history lists
        if track_history:  
            self.cost_history.append(self.function_values)
            self.cost_history_mean.append(self.cloud_mean_func) 
            self.cloud_history.append(self.cloud.T + 0)
            self.cloud_history_mean.append(self.cloud_mean)
            self.cloud_var_history.append(self.cloud_var)
            if "gradient_descent" not in method: self.gradS_history.append(gradS(self.cloud,kernel_a))

        if(verbose):
            print("Iteration: {:05} -  Function value at cloud mean: {:.5f} - Cloud variance: {:.5f}".format(i, self.cloud_mean_func, self.cloud_var))

        if self.cloud_var < var_epsilon: 
            print("Convergence achieved - Particles are localized")
            break

    if i == (max_iterations - 1): print("Maximum amount of iterations reached")
        
    self.function_evaluations = self.elapsed_iterations*self.N* (1+ 2 *int(gradient_required))
    print("\nFunction value at cloud mean: " + str(self.cloud_mean_func))
    print("Function value evaluated {:01} times".format(self.function_evaluations))





@timerfunc
def train_fmin(self, max_iterations, var_epsilon, learning_rate, method, N_sample, kernel_a, alpha_init, alpha_rate, beta, gamma_init, gamma_rate, verbose, track_history):

    from mlswarm.optimizers import update_swarm_func, update_swarm_derivative_free_func, update_gd_func, update_swarm_func_nesterov, update_swarm_derivative_free_func_nesterov, update_gd_func_nesterov
    from mlswarm.utils import gradient, gradS
  
    #cheching if it is necessary to compute gradients
    gradient_required = (method in ["gradient_descent","gradient_descent_nesterov", "swarm", "swarm_nesterov"]) #or time_step_control

    #define necessary history variables for nesterov method
    if "nesterov" in method:
        lamb = 0
        yf = 0

    #init of history lists
    if track_history:
        self.cost_history = []
        self.cost_history_mean = []
        self.cloud_history = []
        self.cloud_history_mean = []
        self.cloud_var_history = []
        self.gradS_history = []

    #compute necessary values
    self.elapsed_iterations = 0  
    gamma = gamma_init  
    alpha = alpha_init
    self.function_values = self.func(self.cloud.T)

    for i in range(max_iterations):
        
        #compute gradients
        if gradient_required:
            gradients = gradient(self.func, self.cloud.T).T

        #update cloud
        if method == "swarm":                       self.cloud, _ = update_swarm_func(self.cloud, gradients, learning_rate, self.N, kernel_a, alpha, beta, gamma)
        elif method == "swarm_derivfree":           self.cloud, _ = update_swarm_derivative_free_func(self.cloud, self.function_values, learning_rate, self.N, kernel_a, alpha, beta, gamma, N_sample)
        elif method == "gradient_descent":          self.cloud, _ = update_gd_func(self.cloud, gradients, learning_rate)
        #elif method == "swarm_nesterov":            cloud, yf, lamb, cloud_var = update_swarm_func_nesterov(cloud, yf, lamb, elapsed_iterations, gradients, learning_rate, N, kernel_a, alpha, beta, gamma)
        #elif method == "swarm_derivfree_nesterov":  cloud, yf, lamb, cloud_var = update_swarm_derivative_free_func_nesterov(cloud, yf, lamb, elapsed_iterations, function_values, learning_rate, N, kernel_a, alpha, beta, gamma)
        #elif method == "gradient_descent_nesterov": cloud, yf, lamb, cloud_var = update_gd_func_nesterov(cloud, yf, lamb, elapsed_iterations, gradients, learning_rate)
        else: raise Exception("No method found")
            

        #compute cloud properties
        self.function_values = self.func(self.cloud.T)
        self.cloud_var = np.mean(get_var(self.cloud))
        self.cloud_mean = np.mean(self.cloud, axis=0)
        self.cloud_mean_func = self.func(self.cloud_mean)

        #update alpha and elapsed_iterations
        alpha += alpha_rate
        gamma += gamma_rate
        self.elapsed_iterations += 1

        #appending results to history lists
        if track_history:  
            self.cost_history.append(self.function_values + 0)
            self.cost_history_mean.append(self.cloud_mean_func + 0) 
            self.cloud_history.append(self.cloud.T + 0)
            self.cloud_history_mean.append(self.cloud_mean + 0)
            self.cloud_var_history.append(self.cloud_var + 0)
            if "gradient_descent" not in method: self.gradS_history.append(gradS(self.cloud,kernel_a))

        if(verbose):
            print("Iteration: {:05} -  Function value at cloud mean: {:.5f} - Cloud variance: {:.5f}".format(i, self.cloud_mean_func, self.cloud_var))

        if self.cloud_var < var_epsilon: 
            print("Convergence achieved - Particles are localized")
            break

    if i == (max_iterations - 1): print("Maximum amount of iterations reached")
        
    self.function_evaluations = self.elapsed_iterations*self.N* (1 + 2 *int(gradient_required))
    print("\nFunction value at cloud mean: " + str(self.cloud_mean_func))
    print("Function value evaluated {:01} times".format(self.function_evaluations))