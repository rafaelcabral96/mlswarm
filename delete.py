from mlswarm.mlswarm import function
import numpy as np
np.random.seed(42)

func_list = {}
track_history = False

f = lambda x: (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2
name = "Beale function"
func = function(f, name)

np.random.seed(42)
func.init_cloud(np.array(np.random.normal(-2,1,(2,50))))

func.minimize(max_iterations = 20000, var_epsilon = 0.0001, 
              learning_rate = 0.0001,
              method = "swarm_derivfree", N_sample = False,
              kernel_a = 0.01, alpha_init = 0, alpha_rate = 0, beta=0, gamma_init=0, gamma_rate = 0,
              verbose = True, track_history = track_history)

func_list[func.name] = func
#func.plot_everything(folder_name = func.name, f_dim = 2, log = True)