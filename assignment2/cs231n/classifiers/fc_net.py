from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization = None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        for k, (i, j) in enumerate(zip([input_dim, *hidden_dims], [*hidden_dims, num_classes])):
            self.params[f'W{k+1}'] = np.random.randn(i, j) * weight_scale
            self.params[f'b{k+1}'] = np.zeros(j)

            if self.normalization and k < self.num_layers-1:
                self.params[f'gamma{k+1}'] = np.ones(j)
                self.params[f'beta{k+1}'] = np.zeros(j)

        '''self.params['W1'] = np.random.randn(input_dim,hidden_dims[0])*weight_scale
        self.params['bias1'] = np.zeros((1,hidden_dims[0]))
        self.params['gamma1'] = np.ones(1)
        self.params['beta1'] = np.zeros(1)
        for k,v in enumerate(hidden_dims,start =1):
          if k == len(hidden_dims):
            self.params['W{}'.format(k+1)] = np.random.randn(hidden_dims[-1],num_classes)*weight_scale
            self.params['bias{}'.format(k+1)] = np.zeros((1,num_classes))
          else:
            self.params['W{}'.format(k+1)] = np.random.randn(hidden_dims[k-1],hidden_dims[k])*weight_scale
            self.params['bias{}'.format(k+1)] = np.zeros((1,hidden_dims[k]))
            self.params['gamma{}'.format(k+1)] = np.ones(1)
            self.params['beta{}'.format(k+1)] = np.zeros(1)
        '''
        pass
      
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        '''for k in range(1,self.num_layers+1):
            if k == (self.num_layers):
              scores = relu.dot(self.params['W{}'.format(k)]) + self.params['bias{}'.format(k)]
            if k == 1:
              out = X.dot(self.params['W{}'.format(k)]) + self.params['bias{}'.format(k)]
              BNout = (out - np.mean(out, axis =1).reshape(-1,1))/ np.std(out, axis = 1).reshape(-1,1)
              BNout = self.params['gamma{}'.format(k)]*BNout + self.params['beta{}'.format(k)]
              relu = np.maximum(0,BNout)
              U = (np.random.rand(*relu.shape) < self.dropout_param['p']) / self.dropout_param['p']
              relu *= U
            else: 
              out = relu.dot(self.params['W{}'.format(k)]) + self.params['bias{}'.format(k)]
              BNout = (out - np.mean(out, axis =1).reshape(-1,1))/ np.std(out, axis = 1).reshape(-1,1)
              BNout = self.params['gamma{}'.format(k)]*BNout + self.params['beta{}'.format(k)]
              relu = np.maximum(0,BNout)
              U = (np.random.rand(*relu.shape) < self.dropout_param['p']) / self.dropout_param['p']
              relu *= U
        '''
        cache = {}
        def generic_forward(X,num_layers):
          for l in range(num_layers):
            keys = [f'W{l+1}', f'b{l+1}', f'gamma{l+1}', f'beta{l+1}']   # list of params
            w, b, gamma, beta = (self.params.get(k, None) for k in keys) # get param vals
            bn = self.bn_params[l] if gamma is not None else None  # bn params if exist
            do = self.dropout_param if self.use_dropout else None  # do params if exist
            if l != num_layers-1:
              X, cache['{}-{}'.format(l+1,1)] = affine_forward(X, w, b)
              if bn is not None: 
                if self.normalization == "layernorm":
                  X, cache['{}-{}'.format(l+1,2)] = layernorm_forward(X, gamma, beta, bn)
                  X, cache['{}-{}'.format(l+1,3)] = relu_forward(X)
                  if self.use_dropout:
                    X, cache['{}-{}'.format(l+1,4)] = dropout_forward(X,do)
                if self.normalization == "batchnorm":
                  X, cache['{}-{}'.format(l+1,2)] = batchnorm_forward(X, gamma, beta, bn)
                  X, cache['{}-{}'.format(l+1,3)] = relu_forward(X)
                  if self.use_dropout:
                    X, cache['{}-{}'.format(l+1,4)] = dropout_forward(X,do)
              else:
                X, cache['{}-{}'.format(l+1,2)] = relu_forward(X)
                if self.use_dropout:
                  X, cache['{}-{}'.format(l+1,3)] = dropout_forward(X,do)
            else:
              X, cache['{}'.format(num_layers)] = affine_forward(X, w, b)
          return X
        scores = generic_forward(X,self.num_layers)
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        def generic_backward(scores,num_layers):
          loss, dout =  softmax_loss(scores, y)
          loss += 0.5 * self.reg * np.sum(np.sum(W**2) for k, W in self.params.items() if 'W' in k)
          dx, grads[f'W{num_layers}'], grads[f'b{num_layers}'] = affine_backward(dout, cache['{}'.format(num_layers)])
          grads[f'W{num_layers}'] += self.reg * self.params[f'W{num_layers}']
          for l in range(num_layers):
            if l == 0:
              continue
            else: 
              if self.normalization is not None:
                if self.normalization == "layernorm":
                  if self.use_dropout: 
                    dx = dropout_backward(dx, cache['{}-{}'.format(num_layers-l,4)])
                    dx = relu_backward(dx, cache['{}-{}'.format(num_layers-l,3)])
                    dx,grads[f'gamma{num_layers-l}'], grads[f'beta{num_layers-l}'] = layernorm_backward(dx, cache['{}-{}'.format(num_layers-l, 2)])
                    dx, grads[f'W{num_layers-l}'], grads[f'b{num_layers-l}'] = affine_backward(dx, cache['{}-{}'.format(num_layers-l, 1)])
                  else: 
                    dx = relu_backward(dx, cache['{}-{}'.format(num_layers-l,3)])
                    dx,grads[f'gamma{num_layers-l}'], grads[f'beta{num_layers-l}'] = layernorm_backward(dx, cache['{}-{}'.format(num_layers-l, 2)])
                    dx, grads[f'W{num_layers-l}'], grads[f'b{num_layers-l}'] = affine_backward(dx, cache['{}-{}'.format(num_layers-l, 1)])
                    grads[f'W{num_layers-l}'] += self.reg * self.params[f'W{num_layers-l}']
                if self.normalization == "batchnorm": 
                  if self.use_dropout: 
                    dx = dropout_backward(dx, cache['{}-{}'.format(num_layers-l,4)])
                    dx = relu_backward(dx, cache['{}-{}'.format(num_layers-l,3)])
                    dx,grads[f'gamma{num_layers-l}'], grads[f'beta{num_layers-l}'] = batchnorm_backward(dx, cache['{}-{}'.format(num_layers-l, 2)])
                    dx, grads[f'W{num_layers-l}'], grads[f'b{num_layers-l}'] = affine_backward(dx, cache['{}-{}'.format(num_layers-l, 1)])
                  else: 
                    dx = relu_backward(dx, cache['{}-{}'.format(num_layers-l,3)])
                    dx,grads[f'gamma{num_layers-l}'], grads[f'beta{num_layers-l}'] = batchnorm_backward(dx, cache['{}-{}'.format(num_layers-l, 2)])
                    dx, grads[f'W{num_layers-l}'], grads[f'b{num_layers-l}'] = affine_backward(dx, cache['{}-{}'.format(num_layers-l, 1)])
                    grads[f'W{num_layers-l}'] += self.reg * self.params[f'W{num_layers-l}']
              if self.normalization is None:
                if self.use_dropout: 
                  dx = dropout_backward(dx, cache['{}-{}'.format(num_layers-l,3)])
                  dx = relu_backward(dx, cache['{}-{}'.format(num_layers-l,2)])
                  dx, grads[f'W{num_layers-l}'], grads[f'b{num_layers-l}'] = affine_backward(dx, cache['{}-{}'.format(num_layers-l, 1)])
                else:
                  dx = relu_backward(dx, cache['{}-{}'.format(num_layers-l,2)])
                  dx, grads[f'W{num_layers-l}'], grads[f'b{num_layers-l}'] = affine_backward(dx, cache['{}-{}'.format(num_layers-l,1)])
                  grads[f'W{num_layers-l}'] += self.reg * self.params[f'W{num_layers-l}']
          return loss, grads
        loss, grads = generic_backward(scores,self.num_layers)

        return loss, grads
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################