import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from time import sleep
import math
import pickle
import pandas as pd



def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer
    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return  1/(1 + np.exp(-z))


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Things to do for preprocessing step:
     - remove features that have the same value for all data points
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - divide the original data set to training, validation and testing set"""

    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary

    train_data = np.concatenate((mat['train0'], mat['train1'],
                                 mat['train2'], mat['train3'],
                                 mat['train4'], mat['train5'],
                                 mat['train6'], mat['train7'],
                                 mat['train8'], mat['train9']), 0)
    train_label = np.concatenate((np.ones((mat['train0'].shape[0], 1), dtype='uint8'),
                                  2 * np.ones((mat['train1'].shape[0], 1), dtype='uint8'),
                                  3 * np.ones((mat['train2'].shape[0], 1), dtype='uint8'),
                                  4 * np.ones((mat['train3'].shape[0], 1), dtype='uint8'),
                                  5 * np.ones((mat['train4'].shape[0], 1), dtype='uint8'),
                                  6 * np.ones((mat['train5'].shape[0], 1), dtype='uint8'),
                                  7 * np.ones((mat['train6'].shape[0], 1), dtype='uint8'),
                                  8 * np.ones((mat['train7'].shape[0], 1), dtype='uint8'),
                                  9 * np.ones((mat['train8'].shape[0], 1), dtype='uint8'),
                                  10 * np.ones((mat['train9'].shape[0], 1), dtype='uint8')), 0)
    test_label = np.concatenate((np.ones((mat['test0'].shape[0], 1), dtype='uint8'),
                                 2 * np.ones((mat['test1'].shape[0], 1), dtype='uint8'),
                                 3 * np.ones((mat['test2'].shape[0], 1), dtype='uint8'),
                                 4 * np.ones((mat['test3'].shape[0], 1), dtype='uint8'),
                                 5 * np.ones((mat['test4'].shape[0], 1), dtype='uint8'),
                                 6 * np.ones((mat['test5'].shape[0], 1), dtype='uint8'),
                                 7 * np.ones((mat['test6'].shape[0], 1), dtype='uint8'),
                                 8 * np.ones((mat['test7'].shape[0], 1), dtype='uint8'),
                                 9 * np.ones((mat['test8'].shape[0], 1), dtype='uint8'),
                                 10 * np.ones((mat['test9'].shape[0], 1), dtype='uint8')), 0)
    test_data = np.concatenate((mat['test0'], mat['test1'],
                                mat['test2'], mat['test3'],
                                mat['test4'], mat['test5'],
                                mat['test6'], mat['test7'],
                                mat['test8'], mat['test9']), 0)
    validation_data = np.concatenate((mat['train0'][:1000], mat['train1'][:1000],
                                 mat['train2'][:1000], mat['train3'][:1000],
                                 mat['train4'][:1000], mat['train5'][:1000],
                                 mat['train6'][:1000], mat['train7'][:1000],
                                 mat['train8'][:1000], mat['train9'][:1000]), 0)
    validation_label = np.concatenate((np.ones((mat['train0'][:1000].shape[0], 1), dtype='uint8'),
                                  2 * np.ones((mat['train1'][:1000].shape[0], 1), dtype='uint8'),
                                  3 * np.ones((mat['train2'][:1000].shape[0], 1), dtype='uint8'),
                                  4 * np.ones((mat['train3'][:1000].shape[0], 1), dtype='uint8'),
                                  5 * np.ones((mat['train4'][:1000].shape[0], 1), dtype='uint8'),
                                  6 * np.ones((mat['train5'][:1000].shape[0], 1), dtype='uint8'),
                                  7 * np.ones((mat['train6'][:1000].shape[0], 1), dtype='uint8'),
                                  8 * np.ones((mat['train7'][:1000].shape[0], 1), dtype='uint8'),
                                  9 * np.ones((mat['train8'][:1000].shape[0], 1), dtype='uint8'),
                                  10 * np.ones((mat['train9'][:1000].shape[0], 1), dtype='uint8')), 0)
    

    duplicatearr = np.all(train_data == train_data[0,:],axis =0)
    dupfeatures  = []
    dupindex = -1;
    for i in duplicatearr:
        dupindex+=1
        if i == True:

            dupfeatures.append(dupindex)

    train_data = np.delete(train_data,dupfeatures, 1)
    validation_data = np.delete(validation_data,dupfeatures,1)
    test_data = np.delete(test_data,dupfeatures,1)

    train_data = np.float64(train_data)
    train_data = train_data / 255

    validation_data = np.float64(validation_data)
    validation_data = validation_data / 255

    test_data = np.float64(test_data)
    test_data = test_data / 255
    print("preprocess done!")

    return train_data, train_label, validation_data, validation_label, test_data, test_label





def feedForwardLayer(layer_weight_matrix, data_with_bias):
    '''
    Calculates W.T* X then return it through the sigmoid function.

    Args:
        weight_matrix: [(d + 1) x M] matrix
        input_matrix: [p x M] matrix

    '''

    #print(data_with_bias.shape)
    #print(layer_weight_matrix.shape)

    O_r = sigmoid(np.dot(data_with_bias, np.transpose(layer_weight_matrix)))

    return O_r


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""
    
    # Your code here
    
    training_data_bias = np.append(data, np.ones((data.shape[0],1)), 1)
    z_j = feedForwardLayer(w1, training_data_bias)
    z_j_bias = np.append(z_j, np.ones((z_j.shape[0],1)), 1)
    out_layer_output = feedForwardLayer(w2, z_j_bias)
    #print(out_layer_output)
    
    labels = np.argmax(out_layer_output, axis=1)

    
    return labels



def errorFunction(output, training_labels):
    '''
    Calculates equation 5 from assignment   
    Args:
        output: The output from forward propogating data_i
        training_labels: The labeled training set
    Returns:
        The value of equeation 5
    
    '''
    log_OL = np.log(output)
    
    # converts the training_labels of size [50,000 x 1]
    # to a matrix of the dummy variables of size [50,000 x 10]
    # aka 1-of-k encoding
    y_dummies = pd.get_dummies(pd.DataFrame(data=training_labels)[0])
    #print(np.unique(training_labels))


    y_dummies = y_dummies.as_matrix().astype(np.float64)

    
    # Just the math
    #print(y_dummies.shape)
    #print(log_OL.shape)
    yL_times_logOL = y_dummies * log_OL
    one_minus_yL = 1 - y_dummies
    log_one_minus_oL = np.log(1 - output)
    
    elem_sum_one_to_n = yL_times_logOL + (one_minus_yL * log_one_minus_oL)
    #print(elem_sum_one_to_n)
    return -np.sum(elem_sum_one_to_n)/len(training_labels)




def equation15(w1, w2, training_data, training_label):
    '''
    Calculates equation 15 from the assignment. Forward feeds the
    data and then calculates the Regularization objective functon.

    Args:
        w1: Weights of hidden layer
        w2: Weights of output layer
        training_data: ...
        training_label: ...
    Returns:
        The value of equeation 15
    
    '''

    # Does the forward feed.
    z_j = feedForwardLayer(w1, training_data)
    z_j_bias = np.append(z_j, np.ones((z_j.shape[0],1)), 1)
    out_layer_output = feedForwardLayer(w2, z_j_bias)
    ol_log = np.log(out_layer_output)

    # 1-k encoding
    y_dummies = pd.get_dummies(pd.DataFrame(data=training_label)[0])
    y_dummies = y_dummies.as_matrix().astype(np.float64) # used in eq 5, 7

    
    learning_coeff = lambdaval/(2*len(training_data))
    J_Reg = learning_coeff * (np.sum(np.square(w1)) + np.sum(np.square(w2)))


    # Equation 5
    J_W1_W2 = errorFunction(out_layer_output, training_label) 

    return (J_W1_W2 + J_Reg), out_layer_output, z_j , y_dummies # equation 15



def equation16_17(w1, w2, z_j, out_layer_output, training_data_bias, y_dummies):
    '''
    Calculates the equations of 16 and 17 from the assignment.

    Args:
        w1: Weights of hidden layer
        w2: Weights of output layer
        z_j: forward feed of input data and out the hidden layer
        out_layer_output: o_:, output from the output layer
        training_data_bias: training_data with a 1's column at the end
    Returns:
        grad_w2: equation 16
        grad_w1: equation 17
    
    '''


    # a little pre process
    z_j = np.append(z_j, np.ones((z_j.shape[0], 1)),1)
    
    delta_L = np.subtract(out_layer_output, y_dummies) # Equation 9
    partial_Ji_W2_Lj = np.dot(np.transpose(delta_L), z_j) # Equation 8
    # Calculating 16
    # Caclulating the gradient for output layer
    lambda_output_weight = np.multiply(lambdaval,w2)
    grad_w2 = np.add(partial_Ji_W2_Lj, lambda_output_weight)/len(out_layer_output) # Eq 16

    # eq 12 (1-z_j)z_j    
    zj_minus_zj_2 = np.subtract(z_j[:, :-1], np.multiply(z_j[:, :-1], z_j[:, :-1])) 
    sum_deltaL_w2_Lj = np.dot(np.subtract(out_layer_output, y_dummies), w2[:,:-1] ) 
    partial_Ji_w1_j = np.multiply(zj_minus_zj_2, sum_deltaL_w2_Lj) 
    partial_Ji_W1_jp = np.dot(partial_Ji_w1_j.T, training_data_bias) # eq 12

    lambda_Wjp = np.multiply(lambdaval, w1)
    grad_w1 = np.add(partial_Ji_W1_jp, lambda_Wjp)/len(training_data_bias)


    return grad_w2, grad_w1




def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    
    #nnPredict(w1, w2, training_data)

    # Your code here
    #
    #
    #

    training_data_bias = np.append(training_data, np.ones((training_data.shape[0],1)), 1)


    obj_val, out_layer_output, z_j, y_dummies  = equation15(w1, w2, training_data_bias,\
                                                                     training_label)


    grad_w2, grad_w1 = equation16_17(w1, w2, z_j, out_layer_output,\
                                                 training_data_bias, y_dummies)


    # Make sure you reshape the gradient matrices to a 1D array. for instance 
    # if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    # obj_grad = np.array([])

    #print(obj_val)
    #print(obj_grad)

    return (obj_val, obj_grad)


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 12

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nnObjFunction(initialWeights, *args)


nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset


print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == (train_label - 1).T).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == (validation_label - 1).T).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == (test_label - 1).T).astype(float))) + '%')











