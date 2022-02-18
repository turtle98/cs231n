from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)
    # initialize the gradient as zero
    #W.shape = (3073,10),  dW = 3073x10
    # compute the loss and the gradient
    num_classes = W.shape[1] #10
    num_train = X.shape[0] #number of train data. #500 
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW.T[j] += X[i]
                dW.T[y[i]] -= X[i] 
    loss
    #X[i] = 1 x 3072, W[j] = 3072x1 
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    dW += 2*reg*W
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1] #10
    num_train = X.shape[0] #number of train data. #500 
    
    # Xshape = 500x3073
    # Wshape = 3073x10 
    '''scores = X.dot(W)
    correct_class_scores = scores[range(num_train),list(y)].reshape(-1,1)
    margins = np.maximum(0,scores-correct_class_scores+1)
    margins[range(num_train),list(y)] = 0
    loss = np.sum(margins)/num_train + 0.5*reg*np.sum(W*W)
    '''
    scores = np.dot(X,W)
    correct_scores = scores[range(num_train), list(y)].reshape(-1,1)
    margin = np.maximum(0,scores-correct_scores+1)
    '''
    !!!!!!np.max와 np.maximum은 다르다!!!!!!
    np.max는 최댓값 하나를 빼오고, np.maximum
    np.max: This function only works on a single input array and 
    finds the value of maximum element in that entire array (returning a scalar).
    np.maximum: this function takes two take two arrays 
    and compute their element-wise maximum.
    '''
    
    margin[range(num_train),list(y)] = 0
    loss = np.sum(margin) / num_train
    loss += reg * 0.5*np.sum(W * W)
    pass
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    coeff = np.zeros(margin.shape)
    coeff[margin > 0] = 1
    coeff[range(num_train),list(y)] = 0
    coeff[range(num_train),list(y)] = -np.sum(coeff, axis=1) 
    dW = X.T.dot(coeff)
    dW /= num_train
    dW += 2*reg*W
    #dW.T[list(y),] -= X[range(num_train),] 
    #range(y)는 snytax 오류가 뜬다!!!!!
    #range()는 integer scalar를 받는다!!! 
    #dW.T[margin > 0] =  # 1x3073 
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
