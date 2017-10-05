# coding = <utf-8>

"""
implement of MLP based on tensorflow

Model Structure:
    Input Layer[0] : X [64*64*3]
    Hidden Layer[1] : 25 units, Relu
    Hidden Layer[2] : 12 units, Relu
    Output Layer[3] : Y [6], Softmax
"""
##======== packages ========##
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot

##======== functions ========##
def initialize_parameters_MLP(seed=1):
    """
    initialization parameters of MLP model (3-layers)
    
    (Input)  
        Layer[0] u[0] = x with shape(x) = [12288,]
    (Hidden) 
        Layer[1] z[1] = w[1]u[0]+b[1] with shape(w[1]) = [25,12288], shape(b[1]) = [25,1]
                 u[1] = activate(z[1])
        Layer[2] z[2] = w[2]u[1]+b[2] with shape(w[2]) = [12,25], shape(b[2]) = [12,1]
                 u[2] = activate(z[2])
    (Output)
        Layer[3] z[3] = w[3]u[2]+b[3] with shape(w[3]) = [6,12], shape(b[3]) = [6,1]
                 u[3] = activate(z[3])
                 y[3] = u[3] with shape(y) = [6,]
    
    @param seed: random seed, default(1)
    @return: initialized parameters list
    """
    
    tf.set_random_seed(seed)                   # so that your "random" numbers match ours
        
    W1 = tf.get_variable("W1", [25,12288], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12,25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [12,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6,12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [6,1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters

def forward_propagation_MLP(X, parameters):
    """
    Implements the forward propagation for the model: 
    
    LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    @param X: input dataset placeholder, of shape (input size, number of examples)
    @param parameters: python dictionary containing your parameters "W1", "b1", "W2", "b2"....
    
    @return Z3: the output of the last lINEAR unit (not SOFTMAX)
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.add(tf.matmul(W1,X),b1)                        # Z1 = W1*X + b1
    U1 = tf.nn.relu(Z1)                                    # U1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2,U1),b2)                       # Z2 = W2*U1 + b2
    U2 = tf.nn.relu(Z2)                                    # U2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3,U2),b3)                       # Z3 = W3*U2 + b3
    # U3 = tf.nn.softmax(Z3)                               # U3 = relu(Z3)
    
    return Z3

def compute_cost_MLP(Z3, Y):
    """
    Computes the cost
    
    @param U3: the output of forward propagation
    @param Y: labels vector placeholder, same shape as U3
    
    @return cost: Tensor of the cost function
    """
    
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    
    return cost

def predict_MLP(X, parameters):
    """
    prediction on test sample X
    
    @param X: sample of shape [12288, m] as m is the samples number
    @param parameters: MLP's parameters
    
    @return: predict labels Y of shape [1, m]
    """
    
    # load parameter
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    # construct computation graph
    x = tf.placeholder("float", [12288, X.shape[1]])    
    z3 = forward_propagation_MLP(x, params)
    Y_pred = tf.nn.softmax(logits=z3, dim=0)
    p = tf.argmax(Y_pred, axis=0)
    
    # create session and run
    with tf.Session() as sess:
        y_pred = sess.run(p, feed_dict = {x: X})

    return y_pred

def model_train_MLP(X_train, Y_train, X_test, Y_test, 
                    learning_rate=0.0001, epochs=1500, minibatch_size=32, 
                    seed=1, verbose=0):
    """
    training a three-layer tensorflow neural network: 
        LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX
    
    @param X_train: training set of shape [12288, m], m is the number of training sample size
    @param Y_train: training label of shape [6, m]
    @param X_test: testing set while training with shape [12288, m_tst], m_tst is the number of test sample size 
    @param Y_test: testing label of shape [6, m_tst]
    
    @param learning_rate: learning rate of the optimization
    @param epochs: number of epochs of training loop
    @param minibatch_size: size of a minibatch
    @param seed: random seed
    @param verbose: 0 to keep silence
                    1 to print the cost of training for 10 epoch
                    2 to print the cost of training & testing for 10 epoch
    
    @return: parameters - after model training 
    @return: cost of train and test if need
    """
    
    tf.set_random_seed(seed)
    ops.reset_default_graph()  # re-run of model without overwriting variables

    n_x, m = X_train.shape
    n_y, _ = Y_train.shape
    train_costs = []  # keep track of the train cost
    test_costs = []  # keep track of the test cost
    
    #---- create the computation graph ----# 
    # create placeholder of shape [n_x, n_y] in computation graph
    X = tf.placeholder(dtype=tf.float32, shape=[n_x,None], name='X')
    Y = tf.placeholder(dtype=tf.float32, shape=[n_y,None], name='Y')
    # parameters initialization
    parameters = initialize_parameters_MLP(seed)
    # forward propagation
    Z3 = forward_propagation_MLP(X, parameters)
    # calculation of cost
    cost = compute_cost_MLP(Z3, Y)
    # construction of optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
                
    init = tf.global_variables_initializer()  # initial all the variables of tf

    #---- Start the session to compute the tensorflow graph ----#
    with tf.Session() as sess:
        sess.run(init)
        
        # training loop
        for epoch in range(epochs):
            epoch_cost = 0.0  # this turn cost
            num_minibatches = int(m/minibatch_size)
            seed += 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)  # subsample
            
            for minibatch in minibatches:
                minibatch_X, minibatch_Y = minibatch
                        
                # run the graph on a minibatch
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
                epoch_cost += minibatch_cost/num_minibatches
                
            # printing the cost every epoch
            if verbose == 1 and epoch % 10 == 0:
                print("epoch %i: train cost %f" % (epoch, epoch_cost))
                train_costs.append(epoch_cost)   
                   
            if verbose == 2 and epoch % 10 == 0:
                # calculating test cost
                Z3_2 = forward_propagation_MLP(X, sess.run(parameters))  # computation graph
                cost_2 = compute_cost_MLP(Z3_2, Y)
                with tf.Session() as sess_2:  # session
                    test_cost = sess_2.run(cost_2, feed_dict={X:X_test, Y:Y_test})  
                               
                print("epoch %i: train cost%f, test cost%f" % (epoch, epoch_cost, test_cost))
                train_costs.append(epoch_cost)  
                test_costs.append(test_cost)
                
        # save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")        

        # get accuracy result
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))    

        return parameters, train_costs, test_costs
    

if __name__ == '__main__':
    ## data loading and pre_processing (scale transform)
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    # Flatten the training and test images
    X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
    X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
    X_train = X_train_flatten/255.
    X_test = X_test_flatten/255.
    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, 6)
    Y_test = convert_to_one_hot(Y_test_orig, 6)
    
    print ("number of training examples = " + str(X_train.shape[1]))
    print ("number of test examples = " + str(X_test.shape[1]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))
    
    ## construct model, training & testing
    lr = 0.001

    parameters, train_costs, test_costs = model_train_MLP(X_train, Y_train, X_test, Y_test, 
                                                          learning_rate=lr, epochs=1000,
                                                          verbose=2)
    
    pickle.dump(parameters, open("parameters",'wb')) 

    ## plot the cost during training and testing
    plt.plot(np.squeeze(train_costs), label='train cost')
    plt.plot(np.squeeze(test_costs), label='test cost')
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("MLP train/test cost \n learning rate =" + str(lr))
    plt.grid(True)
    plt.legend()
    plt.show()

    ## result visualizing (display 10 images)
    parameters = pickle.load(open("parameters",'rb'))
    y_pred = predict_MLP(X_test, parameters)  # predict

    n = 10  
    img_x = X_test_orig[0:n]
    img_y = Y_test_orig[0,0:n]
    for i in range(n):  # display
        plt.subplot(2, 5, i+1); 
        plt.axis("off")
        plt.imshow(img_x[i])
        plt.title('label=%d\npred=%d' % (img_y[i],y_pred[i]))   
    plt.show()    
    
    print(" - PY131 -")