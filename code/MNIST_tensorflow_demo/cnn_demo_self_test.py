"""
@author: PnYuan
@summary: here we test our implemented CNN on real hand-writing number
"""

#========== packages ==========#
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import scipy.misc  # for images to matrix
from scipy import ndimage
from cnn_demo import lenet_5_forward_propagation

#========== paths ==========#
model_path = './model/model.ckpt'
data_path = '../data/self_test_data/'

#========== main process ==========#
if __name__ == "__main__":
    
    #--- re-show the model ---#
    saver = tf.train.import_meta_graph("./model/model.ckpt.meta") 
    
    with tf.Session() as sess:  
              
        #----- load previous model-----#
        saver.restore(sess, model_path)
        
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name("X:0")
        Y = graph.get_tensor_by_name("Y:0")
        Y_conv = graph.get_tensor_by_name("Softmax:0")
        
        #----- load my own data-----#
        f1 = plt.figure(1)
        for i in range(10):
            image_path = data_path + str(i+1) + ".png"
            image = np.array(ndimage.imread(image_path, flatten=True))  # mode = 'L' for convert RGB to Gray-Scale
            x = scipy.misc.imresize(image, size=(28, 28))
            x = (255 - x)/ 255. # normalization for approaching mnist's distribution
            x = x.reshape((1, 28 * 28 * 1))
            y = np.zeros((1,10))  # this y has no mean
            
            #----- test -----#
            y_pred = np.argmax(sess.run(Y_conv, feed_dict={X: x, Y: y}), 1)
            
            #----- display -----#
            ax = plt.subplot(2, 5, i+1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.title.set_text("pred: %d" % y_pred[0])
            # plt.imshow(x[0].reshape(28, 28), cmap = 'binary')  # display in gray
            plt.imshow(ndimage.imread(image_path, flatten = False))  # display in color
        plt.show()
        
    print(" ~ PnYuan - PY131 ~ ")