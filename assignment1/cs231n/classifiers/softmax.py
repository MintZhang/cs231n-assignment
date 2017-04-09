import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train=X.shape[0]
  num_class=W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  data_loss=0.0
  scores=X.dot(W)
  pro=np.zeros_like(scores)
  pros=np.zeros_like(scores)
  for i in xrange(num_train):
  	exp_y_row=np.exp(scores[i,:])
  	pro[i,:]=exp_y_row/np.sum(exp_y_row,keepdims=True)
  	pros[i,:]=-np.log(pro[i,:])
  	data_loss+=pros[i,y[i]]
  	Xi=X.T[:,i]
  	proi=pro[i,:]
  	proi[y[i]]-=1
 	dW+=Xi[:,np.newaxis].dot(proi[np.newaxis,:])

  data_loss/=num_train
  reg_loss=0.5*reg*np.sum(W**2)
  loss=data_loss+reg_loss
  dW/=num_train
  dW+=reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train=X.shape[0]
  num_class=W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores=X.dot(W)
  exp_scores=np.exp(scores)
  pros=exp_scores/np.sum(exp_scores,axis=1,keepdims=True)
  correct_pros=pros[range(num_train),y]
  Li_loss=-np.log(correct_pros)
  data_loss=np.sum(Li_loss)/num_train
  reg_loss=0.5*reg*np.sum(W**2)
  loss=data_loss+reg_loss

  dscores=pros
  dscores[range(num_train),y]-=1

  dW=X.T.dot(dscores)/num_train
  dW+=reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

