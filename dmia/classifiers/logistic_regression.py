import numpy as np
from scipy import sparse


class LogisticRegression:
    def __init__(self):
        self.w = None
        self.loss_history = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this classifier using stochastic gradient descent.

        Inputs:
        - X: N x D array of training data. Each training point is a D-dimensional
             column.
        - y: 1-dimensional array of length N with labels 0-1, for 2 classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        # Add a column of ones to X for the bias sake.
        X = LogisticRegression.append_biases(X)
        num_train, dim = X.shape
        print(f"LOG num_train {num_train} dim {dim}")
        if self.w is None:
            # lazily initialize weights
            self.w = np.random.randn(dim) * 0.01

        # Run stochastic gradient descent to optimize W
        self.loss_history = []
        for it in range(num_iters):
            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            print(f"LOG TRAIN STEP 1")
            indexes = np.random.choice(num_train, batch_size)
            X_batch = X[indexes]
            y_batch = y[indexes]
            print(f"LOG X_batch.shape {X_batch.shape} ---- y_batch.shape  {y_batch.shape}")
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # evaluate loss and gradient
            loss, gradW = self.loss(X_batch, y_batch, reg)
            self.loss_history.append(loss)
            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################


            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return self

    def predict_proba(self, X, append_bias=False):
        """
        Use the trained weights of this linear classifier to predict probabilities for
        data points.

        Inputs:
        - X: N x D array of data. Each row is a D-dimensional point.
        - append_bias: bool. Whether to append bias before predicting or not.

        Returns:
        - y_proba: Probabilities of classes for the data in X. y_pred is a 2-dimensional
          array with a shape (N, 2), and each row is a distribution of classes [prob_class_0, prob_class_1].
        """
        if append_bias:
            X = LogisticRegression.append_biases(X)
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the probabilities of classes in y_proba.   #
        # Hint: It might be helpful to use np.vstack and np.sum                   #
        ###########################################################################



        ###########################################################################
        #                           END OF YOUR CODE                              #
        print(f"LOG predict_proba self.w[1:].shape {self.w[1:].shape} X[1].T.shape {X[1].T.shape}")
        # ????? y_proba это сигмоид от суммы произведений w и X(нужно учитывать Xi возможно форма весов неверна)
        array_X = X.toarray()
        x1_array = array_X[1]
        print(f"LOG predict_proba  x1_array.shape  {x1_array.shape} \n type {type(x1_array)} \n self.w {self.w.shape} "
              f"\n type {type(self.w)} \n")
        tmp = self.w + x1_array
        print(f"LOG predict_proba tmp {tmp} \n tmp.shape {tmp.shape}")
        ###########################################################################
        return y_proba

    def predict(self, X):
        """
        Use the ```predict_proba``` method to predict labels for data points.

        Inputs:
        - X: N x D array of training data. Each column is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """

        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        y_proba = self.predict_proba(X, append_bias=False)  # ????? CHANGE
        y_pred = ...

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """Logistic Regression loss function
        Inputs:
        - X: N x D array of data. Data are D-dimensional rows
        - y: 1-dimensional array of length N with labels 0-1, for 2 classes
        Returns:
        a tuple of:
        - loss as single float
        - gradient with respect to weights w; an array of same shape as w
        """
        dw = np.zeros_like(self.w)  # initialize the gradient as zero
        loss = 0   # ????? матожидание между y данными и y предсказанным
        # Compute loss and gradient. Your code should not contain python loops.

        # Right now the loss is a sum over all training examples, but we want it
        # to be an average instead so we divide by num_train.
        # Note that the same thing must be done with gradient.

        #  ????? корректировка весов -  w = w - a(a - y)x,  где a = sigmoid(w^T*x)
        #  ????? Log loss  https://www.helenkapatsa.ru/logharifmichieskaia-potieria/?ysclid=l4xru2cvju431994983
        print(f"LOG call loss {X_batch}  type {type(X_batch)}")
        loss = y_batch - self.predict(X_batch)
        # Add regularization to the loss and gradient.
        # Note that you have to exclude bias term in regularization.


        return loss, dw

    @staticmethod
    def append_biases(X):
        print(f"LOG call append_biases {X}  type {type(X)}")
        return sparse.hstack((X, np.ones(X.shape[0])[:, np.newaxis])).tocsr()
