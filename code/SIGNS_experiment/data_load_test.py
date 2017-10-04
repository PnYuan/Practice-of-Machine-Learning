# coding = <utf-8>

"""
here is an example for SIGNS data load and visualize
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt

def load_dataset():
    """
    SIGNS data loading
    
    @return: train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
    """
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

if __name__ == '__main__':
    # load data
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    
    # display some (10 images)
    n = 10 
    img_x = X_test_orig[0:n]
    img_y = Y_test_orig[0,0:n]
    # display
    for i in range(n): 
        plt.subplot(2, 5, i+1); 
        plt.axis("off")
        plt.imshow(img_x[i])
        plt.title('label=%d' % (img_y[i]))   
    plt.show()
    
    print(" - PY131 - ")