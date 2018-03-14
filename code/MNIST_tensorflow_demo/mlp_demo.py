"""
@author: PnYuan
@summary: implementing DNN (multi-layer perceptron) for MNIST based on tensorflow
"""

#========== packages ==========#
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data  # for data loading

import numpy as np
import math
import time

#========== file paths ==========#
mnist_data_path = '../data/mnist_data/'
param_path = './param'
logs_path = 'C:\\Users\\Peng\\Desktop\\logs'

#========== functions ==========#
'''data shuffle and mini-batches partition'''
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    @description: Creates a list of random minibatches from (X, Y)
    
    @param X: input data, of shape (input size, number of examples)
    @param Y: output data, of shape (output size, number of examples)
    @param mini_batch_size: size of the mini-batches, integer
    @param seed: to keep random minibatches are the same in different tuning.

    @return: mini_batches: list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

'''(MLP) parameter initial'''
def mlp_param_init(dim, scheme = 'zero'):
    """
    @note: Initializes parameters to build a multi-layer perceptron with tensorflow.
        The shapes are:
            W1: [n1, n_x]
            B1: [n1, 1]
            W2: [n2, n1]
            B2: [n2, 1]
            ...
            Wl: [n_y, nl-1]
            Bl: [n_y, 1]
        
    @param dim: the number of unit in each level -- dim = [n_x, n1, n2, ..., n(l-1), n_y]    
    @param scheme: the initial scheme of Weight, including {'zero', 'xavier'}
    
    @return: parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    parameters = {}
    l = len(dim)  # the layers' count
    
    # parameter initializing (using xavier_initializer for weight)
    # (from 0 - input to l-1 - output)
    for i in range(1, l):
        if scheme == 'xavier':
            parameters['W'+str(i)] = tf.get_variable('W'+str(i), [dim[i], dim[i-1]], \
                                                     initializer = tf.contrib.layers.xavier_initializer())
        else:
            parameters['W'+str(i)] = tf.get_variable('W'+str(i), [dim[i], dim[i-1]], \
                                                     initializer = tf.zeros_initializer())     
        parameters['B'+str(i)] = tf.get_variable('B'+str(i), [dim[i], 1], \
                                                 initializer = tf.zeros_initializer())

    return parameters

'''(MLP) forward propagation'''
def mlp_forward_propagation_with_dropout(X, parameters, keep_prob):
    """
    @note: Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU ... -> LINEAR -> (SOFTMAX)
    
    @param X: input dataset placeholder, of shape (input size, number of examples)
    @param parameters: python dictionary containing your parameters "W1", "B1", "W2", "B2", ...
                       the shapes are given in initialize_parameters.
    @para keep_prob: the list of keep_prob in hidden layer for dropout (this is a placeholder)
    
    @return: Z_l the output of the last LINEAR unit
    """
    
    l = int(len(parameters)/2)  # the number of layer
    
    # for hidden layer(linear-->relu)
    A = X
    for i in range(1, l):
        W = parameters['W'+str(i)]
        B = parameters['B'+str(i)]
        Z = tf.add(tf.matmul(W, A), B)  # Z = np.dot(W, X) + B
        # add dropout
        Z = tf.nn.dropout(Z, keep_prob[i])
        
        A = tf.nn.relu(Z)               # A = relu(Z)
    
    # for output layer(linear-->softmax) softmax is canceled here
    W = parameters['W'+str(l)]
    B = parameters['B'+str(l)]
    Z_l = tf.add(tf.matmul(W, A), B)

    return Z_l

'''(MLP) cost function''' 
def mlp_compute_cost(Z_l, Y):
    """
    @note: Computes the cost
    
    @param Z_l: output of forward propagation (output of the last LINEAR unit), of shape (ny, number of examples)
    @param Y: "true" labels vector placeholder, same shape as Z3
    
    @return: cost, Tensor of the cost function
    """
    logits = tf.transpose(Z_l)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    return cost

#========== main process ==========#
if __name__ == "__main__":
    
    #--- load data ---#
    mnist = input_data.read_data_sets(mnist_data_path, one_hot=True)  # using one-hot for output
    X_train_org, Y_train_org = mnist.train.images, mnist.train.labels
    X_valid_org, Y_valid_org = mnist.validation.images, mnist.validation.labels
    X_test_org,  Y_test_org  = mnist.test.images, mnist.test.labels
    
    #--- get the shape of data ---#
    m_train, n_x = X_train_org.shape
    _, n_y       = Y_train_org.shape
    m_valid, _   = X_valid_org.shape
    m_test, _    = X_test_org.shape
    
    #--- reshape the data (for vectorization) ---#
    X_train = X_train_org.transpose()
    Y_train = Y_train_org.transpose()
    X_valid = X_valid_org.transpose()
    Y_valid = Y_valid_org.transpose()
    X_test = X_test_org.transpose()
    Y_test = Y_test_org.transpose()
    
    #--- build the model ---#
    X = tf.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")

    # the list of unit number ( input layer - hidden layer - output layer )
    dim = [784, 50, 200, 50, 10]
    
    # for dropout
    keep_prob = tf.placeholder(tf.float32, [len(dim),], name='keep_prob')
    keep_1 = [1, 0.7, 0.5, 0.7, 1]  # for trian
    keep_2 = [1, 1, 1, 1, 1]  # for test
    
    # initial parameters (using Xavier initial scheme for weight)
    params = mlp_param_init(dim, scheme='xavier')

    # build forward computation graph
    Z_l = mlp_forward_propagation_with_dropout(X, params, keep_prob)
    Y_l = tf.nn.softmax(Z_l, dim = 0)
    Y_pred = tf.argmax(Y_l)
    
    # cost computation (linked to tensorboard)
    with tf.name_scope('training'):
        cost = mlp_compute_cost(Z_l, Y)
        tf.summary.scalar('cost', cost)
        
    # accuracy computation (linked to tensorboard)
    with tf.name_scope('training'):
        correct_prediction = tf.equal(tf.argmax(Y, axis=0), tf.argmax(Y_l, axis=0))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) 
        tf.summary.scalar('accuracy', accuracy)
    
    #--- train the model ---#
    # hyper-parameter
    learning_rate = 0.001
    num_epochs = 1000
    minibatch_size = 512
    
    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # initial the graph and session
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()  
    sess.run(init)
    
    # linked to tensorboard
    merged = tf.summary.merge_all()  # merge all data for dashborad
    writer = tf.summary.FileWriter(logs_path, sess.graph)  # write training data in "logs" file

    # iterations
    start = time.time()  # time count

    for i in range( int(num_epochs*m_train/minibatch_size) ):
        batch_xs, batch_ys = mnist.train.next_batch(minibatch_size, shuffle = True)  # using mini-batch
        _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: batch_xs.transpose(), Y: batch_ys.transpose(),  keep_prob: keep_1})
    
        if(i % 20 == 0):  # wirte logs
            result = sess.run(merged, feed_dict={X: X_train, Y: Y_train, keep_prob: keep_2})
            writer.add_summary(result, i)

        if(i % 100 == 0):  # print taining process
            print("iteration %d training minibatch_cost %f" % (i, minibatch_cost))

    end = time.time()
    print("Time consume", end-start)
    
    print("Parameters have been trained!")
    print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train, keep_prob: keep_2}))
    print("Valid Accuracy:", accuracy.eval({X: X_valid, Y: Y_valid, keep_prob: keep_2}))
    print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test, keep_prob: keep_2}))
    
    writer.close()
    sess.close()
    
    print(" ~ PnYuan - PY131 ~ ")