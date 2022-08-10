import numpy as np
import random
from scipy import sparse


class LogisticRegression:
    def __init__(self):
        self.w = None
        self.loss_history = None

    def train(self, X, y, learning_rate=100, reg=1e-5, num_iters=10,
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
        X = LogisticRegression.append_biases(X)  # Add a column of ones to X for the bias sake.
        num_train, dim = X.shape
        if self.w is None:
            # lazily initialize weights
            self.w = np.random.randn(dim) * 0.01
        self.loss_history = []
        # Run stochastic gradient descent to optimize W
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
            indexes = np.random.choice(num_train, batch_size)
            X_batch = X[indexes]
            y_batch = y[indexes]
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # evaluate loss and gradient
            loss, gradW = self.loss(X_batch, y_batch, reg)
            print(f"Log train {loss} --- {gradW}")
            self.loss_history.append(loss)
            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            self.w -= learning_rate * gradW
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            if verbose and it % 1 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
        y_pred = X @ self.w > 0
        print(f"Log y_pred {y_pred} \n {y_pred.shape}")
        y_pred2 = X @ self.w
        print(f"Log y_pred2 {y_pred2} \n {y_pred2.shape}")
        return self

    def sigma(self, z):
        return 1 / (1 + np.exp(-z))

    def calculate_sigmoid(self, Xi):
        return self.sigma(np.sum(self.w * Xi.toarray()))

    def grad(self, y, X):
        n = y.shape[0]
        X = X.toarray()
        print(f"grad {X.shape} --- {y.shape} --- {self.w})")
        return 1 / n * X.transpose() @ (self.sigma(X @ self.w) - y)

    def L(self, y, X):
        n = y.shape[0]
        return -1 / (n) * np.sum(y * np.log(self.sigma(X @ self.w)) + (1 - y) * np.log(1 - self.sigma(X @ self.w)))

    def calculate_negative_prob(self, prob):
        return [prob, 1 - prob]

    def chouse_class(self, variants):
        if variants[0] >= 0.8:
            return 0
        else:
            return 1

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
        # array_X = X.toarray()
        probs = list(map(self.calculate_sigmoid, X))
        y_proba = np.array(list(map(self.calculate_negative_prob, probs)))
        ###########################################################################
        return y_proba

    def predict(self, X):
        print(f"predict {X.shape}")
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
        y_proba = self.predict_proba(X, append_bias=True)
        y_pred = np.array(list(map(self.chouse_class, y_proba)))
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
        loss = 0
        # Compute loss and gradient. Your code should not contain python loops.

        # Right now the loss is a sum over all training examples, but we want it
        # to be an average instead so we divide by num_train.
        # Note that the same thing must be done with gradient.
        y_predict = np.array(list(map(self.calculate_sigmoid, X_batch)))

        loss = self.L(y_batch, X_batch)
        dw = self.grad(y_batch, X_batch)
       # loss += reg * np.linalg.norm(self.w)  # Добавление регуляризации
        # Add regularization to the loss and gradient.


        return loss, dw

    @staticmethod
    def append_biases(X):
        return sparse.hstack((X, np.ones(X.shape[0])[:, np.newaxis])).tocsr()
