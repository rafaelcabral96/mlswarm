import numpy as np
from mlswarm.neural_networks import init_layers, full_forward_propagation, full_backward_propagation, get_cost_func, get_cost_func_deriv, get_accuracy_value
from mlswarm.utils import timerfunc, get_mean, get_var, gradient, gradS, flatten_weights, flatten_weights_gradients, unflatten_weights
from mlswarm.optimizers import  update_cloud_derivative_free, update_cloud, nesterov, adaptive_step_size


@timerfunc
def f_minimize(self, max_iterations, epsilon, var_epsilon, lr, method, N_sample, kernel_a, alpha_init, alpha_rate, beta, gamma_init, gamma_rate, verbose, track_history):
  
    #checking if it is necessary to compute gradients
    gradient_required = self.train_method in ["gradient_descent","gradient"]

    #define necessary history variables for nesterov method and euler adaptive
    lamb = 0
    yf = 0
    update = 0

    #compute several initial values
    self.function_values = self.func(self.cloud.T)
    self.cloud_mean_func = self.func(self.cloud_mean)
    self.elapsed_iterations = 0  
    gamma = gamma_init  
    alpha = alpha_init
    init_var = self.cloud_var


    #start cloud update cycle
    for i in range(max_iterations):
        
        update_old = update

        #rescale gamma term
        gamma = gamma * np.mean(self.cloud_var)

        #compute update and cloud variance-------------------------------
        if method == 'gradient':
            gradients = gradient(self.func, self.cloud.T).T
            update, cloud_var = update_cloud(self.cloud, gradients, lr, self.N, kernel_a, alpha, beta, gamma) 
        
        elif method == 'gradient_descent': 
            update = gradient(self.func, self.cloud.T).T
            cloud_var = 100

        elif method == 'gradient_free':
            update, cloud_var = update_cloud_derivative_free(self,  self.cloud, self.function_values, lr, self.N, kernel_a, alpha, beta, gamma)  

        else:
            raise Exception("No method found")


       #update cloud based on chosen algorithm---------------------------
        if self.algorithm == 'euler':
            self.cloud -= lr * update

        elif self.algorithm == 'nesterov':
            self.cloud, yf, lamb = nesterov(self.cloud, yf, lamb, self.elapsed_iterations, lr, update)

        elif self.algorithm == 'euler_adaptive':
            lr = adaptive_step_size(lr, i, update, update_old)
            self.cloud -= lr * update

        elif self.algorithm == 'nesterov_adaptive':
            
            if self.elapsed_iterations == 0:
                lamb = np.ones(update.shape[0])
                yf = self.cloud
            else:
                lamb = (1.0 + np.sqrt(1.0 + 4.0*lamb_prev**2))/2.0

                #nesterov_restart
                lamb[nesterov_restart] = 1
                yf[np.repeat(nesterov_restart,update.shape[1]).reshape(update.shape)] = self.cloud[np.repeat(nesterov_restart,update.shape[1]).reshape(update.shape)]
            
                if self.elapsed_iterations % 100 == 0:
                    lamb = np.ones(update.shape[0])
                    yf = self.cloud

            lamb_next = (1.0 + np.sqrt(1.0 + 4.0*lamb**2))/2.0

            g_const = (1.0 - lamb) / lamb_next
            yfnext = self.cloud - lr * update
            self.cloud = np.einsum('i,ij -> ij',(1.0 - g_const),yfnext) + np.einsum('i,ij ->ij', g_const, yf)

            yf = yfnext
            lamb_prev = lamb

            new_function_values = self.func(self.cloud.T)
            nesterov_restart = new_function_values > self.function_values
     
        else:
            raise Exception("Algorithm not found")
        
        #compute new cloud properties
        self.function_values = self.func(self.cloud.T)
        self.cloud_var = np.mean(get_var(self.cloud)) 
        self.cloud_mean = np.mean(self.cloud, axis=0)
        old_cloud_mean_func = self.cloud_mean_func
        self.cloud_mean_func = self.func(self.cloud_mean)

        #update best particle position, function value
        new_min = np.nanmin(self.function_values) 
        if new_min < self.best_function_value:
            self.best_function_value = new_min + 0
            self.best_particle  = self.cloud[np.nanargmin(self.function_values)] + 0

        #update alpha and elapsed_iterations
        alpha += alpha_rate
        gamma += gamma_rate
        self.elapsed_iterations += 1

        #appending results to history lists 
        if track_history:  
            self.cost_history.append(self.function_values + 0)
            self.cloud_history.append(self.cloud.T + 0)
            self.cost_history_mean.append(self.cloud_mean_func + 0) 
            self.cloud_history_mean.append(self.cloud_mean + 0)
            self.cloud_var_history.append(self.cloud_var + 0)
            if "gradient_descent" not in method: self.gradS_history.append(gradS(self.cloud,kernel_a))

        #print basic cloud information
        if(verbose):
            print("Iteration: {:05} -  Function value at cloud mean: {:.5f} - Cloud variance: {:.5f}".format(i, self.cloud_mean_func, self.cloud_var))

        #check if terminal conditions are met
        if (np.abs(old_cloud_mean_func - self.cloud_mean_func) < epsilon) & (i > 2):
            print("Convergence achieved - Function value at cloud mean stabilized")
            break

        if self.cloud_var < var_epsilon: 
            print("Convergence achieved - Particles are localized")
            break

        #check if cloud exploded
        if self.cloud_var > 10000*init_var:
            print("Failed convergence - Cloud variance too high")
            break

    if i == (max_iterations - 1): print("Maximum amount of iterations reached")
        
    self.function_evaluations = self.elapsed_iterations*self.N* (1 + 2 *int(gradient_required))


@timerfunc
def train_nn(self, X, Y,  method, max_epochs, n_batches,  batch_size, lr, 
              cost_type, kernel_a, alpha_init, alpha_rate, beta, gamma_init, verbose, var_epsilon):

    #check if backprogation is requireself, X, Y, method, algorithm, max_epochs, n_batches, batch_size, learning_rate, cost_type, kernel_a,  alpha_init, alpha_rate, beta, gamma, verbose, var_epsilond
    gradient_required = (method in ["gradient_descent","gradient"])

    #define necessary history variables for nesterov method and euler adaptive
    lamb = 0
    yf = 0
    update = 0

    # initiation of lists storing the cost history 
    alpha = alpha_init
    gamma = gamma_init
    self.elapsed_iterations = 0   
    init_var = self.cloud_var

    #get cost function 
    cost_func = get_cost_func(cost_type)

    #get derivative of of cost function
    if gradient_required:
        cost_func_deriv = get_cost_func_deriv(cost_type) 

    print("\nTraining started...")   

    for i in range(max_epochs):

        for batch in range(n_batches):
   
            update_old = update

            start = batch*batch_size
            end = start + batch_size

            Y_hat = []
            costs = []
            cache = []
            grads = []

            #cycle all particles
            for j in range(self.N):  

                # step forward
                Y_hat_temp, cache_temp = full_forward_propagation(X[:,start:end], self.cloud[j], self.architecture)
                Y_hat.append(Y_hat_temp)
                cache.append(cache_temp)
                
                # calculating cost and saving it to history
                costj = cost_func(Y_hat[j], Y[:,start:end]) 
                costs.append(costj)      

                # step backward - calculating gradient
                if gradient_required:
                    gradsj =  full_backward_propagation(Y_hat[j], Y[:,start:end], cache[j], self.cloud[j], self.architecture, cost_func_deriv)
                    grads.append(gradsj)  
           
            costs = np.squeeze(np.array(costs))

            #update best neural_network, cost
            new_min = np.nanmin(costs) 
            if new_min < self.best_function_value:
                self.best_function_value = new_min
                self.best_particle_nn = self.cloud[np.nanargmin(costs)]
                self.best_particle = flatten_weights([self.best_particle_nn],1)[0]

            #compute update and cloud variance--------------------------------------
            if method == 'gradient':
                cloudf, gradientsf, nn_shape, weight_names = flatten_weights_gradients(self.cloud, grads, self.N)
                update, var = update_cloud(cloudf, gradientsf, lr, N, kernel_a, alpha, beta, gamma) 

            elif method == 'gradient_descent':
                cloudf, update, nn_shape, weight_names = flatten_weights_gradients(self.cloud, grads, self.N)
                var = 0

            elif method == 'gradient_free':
                cloudf, nn_shape, weight_names = flatten_weights(self.cloud, self.N)
                update, var = update_cloud_derivative_free(self,cloudf, costs, lr, self.N, kernel_a, alpha, beta, gamma)

            else:
                raise Exception("Train method not found")


           #update cloud based on chosen algorithm---------------------------
            if self.algorithm == 'euler':
                cloudf -= lr * update
            
            elif self.algorithm == 'nesterov':
                cloudf, yf, lamb = nesterov(cloudf, yf, lamb, self.elapsed_iterations, lr, update)
            
            elif self.algorithm == 'euler_adaptive':
                theta = 0.2
                lr_old = lr
                if i == 0:
                    lr = lr_old
                else:
                    lr = 2 * lr_old * theta * np.sum(np.abs(update))/np.sum(np.abs((update - update_old)))
                    lr = np.min((lr,1))
                    lr = np.max((lr,0.00001))
                cloudf -= lr * update
            
            elif self.algorithm == 'nesterov_adaptive':

                if self.elapsed_iterations == 0:
                    lamb = np.ones(update.shape[0])
                    yf = self.cloud
                else:
                    lamb = (1.0 + np.sqrt(1.0 + 4.0*lamb_prev**2))/2.0

                    #nesterov_restart
                    lamb[nesterov_restart] = 1
                    yf[np.repeat(nesterov_restart,update.shape[1]).reshape(update.shape)] = self.cloud[np.repeat(nesterov_restart,update.shape[1]).reshape(update.shape)]
                
                    if self.elapsed_iterations % 100 == 0:
                        lamb = np.ones(update.shape[0])
                        yf = self.cloud

                lamb_next = (1.0 + np.sqrt(1.0 + 4.0*lamb**2))/2.0

                g_const = (1.0 - lamb) / lamb_next
                yfnext = self.cloud - lr * update
                self.cloud = np.einsum('i,ij -> ij',(1.0 - g_const),yfnext) + np.einsum('i,ij ->ij', g_const, yf)

                yf = yfnext
                lamb_prev = lamb

                new_function_values = self.func(self.cloud.T)
                nesterov_restart = new_function_values > self.function_values

            else:
                raise Exception("No algorithm found")

            #restore NN weight shapes 
            self.cloud = unflatten_weights(cloudf,nn_shape,weight_names,self.N)

            self.elapsed_iterations += 1

        #end of epoch----------------
        self.cloud_mean = get_mean(self.cloud)
        self.cloud_var = np.mean(var)
        self.cost_history.append(costs)
        self.cloud_var_history.append(self.cloud_var)

        Y_hat_mean, _ = full_forward_propagation(X[:,start:end], self.cloud_mean, self.architecture)
        cost_mean = cost_func(Y_hat_mean, Y[:,start:end])
        self.cost_history_mean.append(cost_mean)

        alpha += alpha_rate
        gamma = gamma * np.mean(self.cloud_var)

        #print basic cloud information
        if(verbose):
            print("Iteration: {:05} - Cloud mean cost: {:.5f} - Cloud variance: {:.5f}".format(i, cost_mean, self.cloud_var))

        #check if terminal conditions are met
        if self.cloud_var < var_epsilon: 
            print("Convergence achieved - particles are localized")
            break

        #check if cloud exploded
        if self.cloud_var > 10000*init_var:
            print("Failed convergence - Cloud variance too high")
            break

    if i == (max_epochs - 1): print("Maximum amount of epochs reached")

    print("\nCost function value at cloud mean: " + str(cost_mean))
    print("Best cost was: " + str(self.best_function_value))
    print("Cost function value (derivative) evaluated {:01} ({:01}) times".format(int(self.elapsed_iterations*self.N),int(gradient_required)*self.elapsed_iterations*self.N))




