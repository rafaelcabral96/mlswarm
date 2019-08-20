import numpy as np
from sklearn.metrics import log_loss, f1_score
from mlswarm.utils import convert_prob_into_class, get_mean


def init_layers(nn_architecture, seed, dispersion_factor, seed_between, dispersion_factor_between):
  
    # random seed initiation
    np.random.seed(seed_within)
    # number of layers in our neural network
    number_of_layers = len(nn_architecture)
    # parameters storage initiation
    params_values = {}
    
    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # we number network layers from 1
        layer_idx = idx + 1
        
        # extracting the number of units in layers
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]
        
        # initiating the values of the W matrix
        # and vector b for subsequent layers
        params_values['W' + str(layer_idx)] = np.random.randn(layer_output_size, layer_input_size) * np.sqrt(dispersion_factor_within/(layer_input_size + layer_output_size)) 
        params_values['b' + str(layer_idx)] = np.random.randn(layer_output_size, 1) * np.sqrt(dispersion_factor/(layer_input_size + layer_output_size)) 
    
    if dispersion_factor_between > 0:
        np.random.seed(seed_between)
        for idx, layer in enumerate(nn_architecture):
            layer_idx = idx + 1
            
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]
            
            params_values['W' + str(layer_idx)] += np.random.randn(layer_output_size, layer_input_size) * np.sqrt(dispersion_factor_within/(layer_input_size + layer_output_size)) 
            params_values['b' + str(layer_idx)] += np.random.randn(layer_output_size, 1) * np.sqrt(dispersion_factor_within/(layer_input_size + layer_output_size)) 

    return params_values


def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    # calculation of the input value for the activation function
    Z_curr = np.dot(W_curr, A_prev) + b_curr
    
    # selection of activation function
    if activation is "relu":
        activation_func = relu
    elif activation is "sigmoid":
        activation_func = sigmoid
    elif activation is "softmax":
        activation_func = softmax
    elif activation is "linear":
        activation_func = linear
    elif activation is "step":
        activation_func = step
    else:
        raise Exception('Non-supported activation function')
        
    # return of calculated activation A and the intermediate Z matrix
    return activation_func(Z_curr), Z_curr

def full_forward_propagation(X, params_values, nn_architecture):
    # creating a temporary memory to store the information needed for a backward step
    memory = {}
    # X vector is the activation for layer 0 
    A_curr = X
    
    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # we number network layers from 1
        layer_idx = idx + 1
        # transfer the activation from the previous iteration
        A_prev = A_curr
        
        # extraction of the activation function for the current layer
        activ_function_curr = layer["activation"]
        # extraction of W for the current layer
        W_curr = params_values["W" + str(layer_idx)]
        # extraction of b for the current layer
        b_curr = params_values["b" + str(layer_idx)]
        # calculation of activation for the current layer
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)
        
        # saving calculated values in the memory
        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr
       
    # return of prediction vector and a dictionary containing intermediate values
    return A_curr, memory


def get_cost_func(type):

    #https://scikit-learn.org/stable/modules/model_evaluation.html#log-loss -> unify cross_entropy_binary and cross_entropy_softmax    
    if type == 'cross_entropy_binary':
        cost_function = lambda Y_hat, Y:  np.squeeze( (-1 / Y_hat.shape[1]) * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T)) )

    elif type == 'cross_entropy_softmax':
        cost_function = lambda Y_hat, Y:  np.squeeze( log_loss(np.transpose(Y),np.transpose(Y_hat)))

    elif type == 'error_classification':
        cost_function = lambda Y_hat, Y:  1.0 - np.mean(Y_hat == Y)  

    elif type == "rmse":    
        cost_function = lambda Y_hat, Y:  np.mean( (Y-Y_hat)**2)      
   
    else:
        raise Exception('No cost type found')

    return cost_function

def get_cost_func_deriv(type):
    
    if type == 'cross_entropy_binary':
        cost_function_deriv = lambda Y_hat, Y:  - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))

    elif type == "rmse":    
        cost_function_deriv = lambda Y_hat, Y:  2 * (Y_hat - Y)      
  
    else:
        raise Exception('No cost type derivative found ')

    return cost_function_deriv


def get_accuracy_value(Y_hat, Y, type):

    if type == 'binary_accuracy': # intended for the output of step functions
        acc = (Y_hat == Y).mean()
    elif type == 'sigmoid_accuracy': # intended for the output of one sigmoid function 
        Y_hat_ = convert_prob_into_class(Y_hat)
        acc = (Y_hat_ == Y).all(axis=1).mean()   
    elif type == 'accuracy_multiclass':  #intended for one hot classification the output of softmax or vector sigmoid
        acc = (np.argmax(Y_hat,1).astype('int32') == np.argmax(Y,1).astype('int32')).mean()
    elif type == 'f1_score':  #intended for one hot classification the output of softmax or vector sigmoid
        acc = f1_score(np.argmax(Y_hat, 1).astype('int32'), np.argmax(Y, 1).astype('int32'), average='macro')
    elif type == "rmse":
        acc =  np.mean( (Y - Y_hat)**2)   
    else:
        raise Exception('No Accuracy type found')

    return acc


def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation):
    # number of examples
    m = A_prev.shape[1]
    
    # selection of activation function
    if activation is "relu":
        backward_activation_func = relu_backward
    elif activation is "sigmoid":
        backward_activation_func = sigmoid_backward
    else:
        raise Exception('Non-supported activation function for backward propagation')
    
    # calculation of the activation function derivative
    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    
    # derivative of the matrix W
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    # derivative of the vector b
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    # derivative of the matrix A_prev
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr



def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture, cost_function_deriv):
    grads_values = {}
    
    # number of examples
    m = Y.shape[1]
    # a hack ensuring the same shape of the prediction vector and labels vector
    Y = Y.reshape(Y_hat.shape)
    
    # initiation of gradient descent algorithm
    dA_prev = cost_function_deriv(Y_hat, Y)

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        # we number network layers from 1
        layer_idx_curr = layer_idx_prev + 1
        # extraction of the activation function for the current layer
        activ_function_curr = layer["activation"]
        
        dA_curr = dA_prev
        
        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]
        
        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]
        
        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)
        
        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr
    
    return grads_values



#ACTIVATION FUNCTIONS---------------------------------------
def linear(Z):
    return Z

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def step(Z):
    step = np.ones(Z.shape)
    step[Z<0] = 0
    return step

def softmax(Z):
    e_Z = np.exp(Z - np.max(Z))
    return e_Z / e_Z.sum(axis=0) 

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0;
    return dZ

