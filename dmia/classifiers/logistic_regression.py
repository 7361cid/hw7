import numpy as np
from scipy import sparse
from scipy.special import expit


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
        # X = LogisticRegression.append_biases(X)  bias term прибавляется при вызове loss
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
            print(f"LOG TRAIN STEP {it}")
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
            print(f"type(learning_rate) {learning_rate} \n {type(learning_rate)} type(gradW) {type(gradW)}")
            self.w = self.w - float(learning_rate) * gradW

            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return self

    def calculate_sigmoid(self, Xi):
        print(f"Log calculate_sigmoid {self.w.shape} --- {Xi.shape}")
        return expit(np.sum(self.w * Xi.toarray()))

    def calculate_negative_prob(self, prob):
        return [prob, 1 - prob]

    def chouse_class(self, variants):
        if variants[0] >= variants[1]:
            return 0
        else:
            return 1

    def calculate_dw(self, Xi, Yi):
        return np.sum(Yi - self.calculate_sigmoid(Xi) * Xi)

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
        print(f"LOG SIZE trouble {X.shape}")
        # array_X = X.toarray()
        probs = list(map(self.calculate_sigmoid, X))
        y_proba = np.array(list(map(self.calculate_negative_prob, probs)))
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
        y_proba = self.predict_proba(X, append_bias=True)
        print(f"Log y_proba {y_proba}")
        y_pred = np.array(list(map(self.chouse_class, y_proba)))
        print(f"Log y_pred {y_pred}")
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
        print(f"Log Before predict X_batch shape {X_batch.shape} self.w.shape {self.w.shape}")
        X_batch = LogisticRegression.append_biases(X_batch)
        y_predict = np.array(list(map(self.calculate_sigmoid, X_batch)))
        x_array = X_batch.toarray()
        print(f"Log after predict X_batch shape {X_batch.shape} self.w.shape {self.w.shape}")
        ones = np.array(list(1 for i in range(y_predict.shape[0])))
        loss = np.mean(- (y_batch * np.log(y_predict) + (ones - y_batch) * np.log(ones - y_predict)))
        loss += reg * np.linalg.norm(self.w)  # Добавление регуляризации
        # Add regularization to the loss and gradient.

        print(f"X_batch shape {X_batch.shape} \n y_batch shape {y_batch.shape}")
        # dw = np.array([self.calculate_dw(X_batch, y_batch) for i in range(X_batch.shape[1])])
        # https://translated.turbopages.org/proxy_u/en-ru.ru.e4bc73ef-62c716fb-60995054-74722d776562/https/www.baeldung.com/cs/gradient-descent-logistic-regression
        # TODO убрать циклы
        print(f"Log self.w.shape {self.w.shape} dw {dw.shape}")
        print(f"Log len(self.w) {len(self.w)} dw {len(dw)}  {len(x_array[0])}")
        # возможно ниже нужно использовать y_predict а не y_batch
        print(f"Log type(y_predict) {type(y_predict)} type(y_batch) {type(y_batch)}  type(x_array) {type(x_array)}")
        print(f"Log y_predict.shape {y_predict.shape} y_batch.shape {y_batch.shape}  x_array.shape {x_array.shape}")
        tmp = y_predict * (y_predict - y_batch)
        tmp = tmp[:, np.newaxis]
        tmp = tmp * x_array
        tmp = tmp.mean(axis=0)
        tmp += reg * np.array(list(map(np.sign, self.w)))  # регуляризация
        print(f"Log tmp {tmp} {type(tmp)} {tmp.shape}")
        dw = tmp
        print(f"Log tmp {tmp} {type(tmp)} {tmp.shape}")
        print(f"Log tmp.shape {tmp.shape} dw.shape {dw.shape}")

        return loss, dw

    @staticmethod
    def append_biases(X):
        return sparse.hstack((X, np.ones(X.shape[0])[:, np.newaxis])).tocsr()
