"""
@author: PnYuan (refer to: http://www.tensorfly.cn/)

@keyword mnist: hand-writting character recognization
@keyword tensorflow: a machine learning development framework
@keyword softmax: a basic classification model in machine learning area.

@summary: here we use softmax to do mnist task.
"""

#========== packages ==========#
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data  # for data loading
import matplotlib.pyplot as plt  # for image showing

#========== data loading ==========#
mnist = input_data.read_data_sets('../data/mnist_data/',one_hot=True)
X_train_org, Y_train_org = mnist.train.images, mnist.train.labels
X_valid_org, Y_valid_org = mnist.validation.images, mnist.validation.labels
X_test_org,  Y_test_org  = mnist.test.images, mnist.test.labels

# check the shape of dataset
print("train set shape: X-", X_train_org.shape, ", Y-", Y_train_org.shape)
print("valid set shape: X-", X_valid_org.shape, ", Y-", Y_valid_org.shape)
print("test set shape: X-", X_test_org.shape, ", Y-", Y_test_org.shape)

# show some of them (1~5)
n = 5
f1 = plt.figure(1)
x_disp = X_train_org[0:n]
y_disp = Y_train_org[0:n]
y_disp = y_disp.argmax(axis=1)  # reverse one-hot coding
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    ax.set_xticks([])  
    ax.set_yticks([])  
    ax.title.set_text("%d" % y_disp[i])
    plt.imshow(x_disp[i].reshape(28,28), cmap='binary')  # display in gray
plt.show()

#========== Softmax Modeling ==========#
x = tf.placeholder("float", [None, 784])  # placeholder of input
y_ = tf.placeholder("float", [None, 10])  # placeholder of label

W = tf.Variable(tf.zeros([784,10]))  # parameters (initial to 0)
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)  # softmax computation graph
y_pred = tf.argmax(y, 1)

# loss (cross-entropy) 
# here we use clip_by_value() to restrict the value between 1e-8 and 1 to avoid log(0)
cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-8,1.0)))

train_step = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cross_entropy)  # using GD

#========== Training ==========#
init = tf.global_variables_initializer()

sess = tf.InteractiveSession()  # initial a session
sess.run(init)

for i in range(1000):  # iterate  for 100 times
    batch_xs, batch_ys = mnist.train.next_batch(100)  # using mini-batch
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
#========== Evaluation ==========#
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))  # use argmax() for decoding one-hot
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # training evaluation

# test on valid set / test set 
print("valid accuracy", sess.run(accuracy, feed_dict={x: X_valid_org, y_: Y_valid_org}))
print("test accuracy", sess.run(accuracy, feed_dict={x: X_test_org, y_: Y_test_org}))

# show some images
y_valid_pred = sess.run(y_pred, feed_dict={x: X_valid_org, y_: Y_valid_org})
y_test_pred = sess.run(y_pred, feed_dict={x: X_test_org, y_: Y_test_org})

# show some of them (1~5)
f2 = plt.figure(2)
x_disp = X_valid_org[0:n]
y_disp = y_valid_pred[0:n]
f2.suptitle("Predict on Valid Set")
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    ax.set_xticks([])  
    ax.set_yticks([])  
    ax.title.set_text("%d" % y_disp[i])
    plt.imshow(x_disp[i].reshape(28,28), cmap='binary')  # display in gray
plt.show()

f3 = plt.figure(3)
x_disp = X_test_org[0:n]
y_disp = y_test_pred[0:n]
f3.suptitle("Predict on Test Set")
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    ax.set_xticks([])  
    ax.set_yticks([])  
    ax.title.set_text("%d" % y_disp[i])
    plt.imshow(x_disp[i].reshape(28,28), cmap='binary')  # display in gray
plt.show()

print(" ~ PnYuan - PY131 ~ ")