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

    def calculate_sigmoid(self, Xi):
       # print(f"Log calculate_sigmoid {Xi.shape} {self.w[1:].shape}")
        return expit(self.w[0] + np.sum(self.w[1:] * Xi))

    def calculate_negative_prob(self, prob):
        return [prob, 1 - prob]

    def chouse_prob(self, variants):
        """
        Должно во3вращать класс или вероятность??????
        """
        if variants[0] >= variants[1]:
            return variants[0]
        else:
            return 1 - variants[0]

    def calculate_dw(self, Xi, Yi):
        #print(f"Log calculate_dw type y_batch {type(y_batch)}\n "
        #      f"\n  type y_predict {type(y_predict)}"
        #      f" type w {type(w)}")
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
        array_X = X.toarray()
        probs = list(map(self.calculate_sigmoid, array_X))
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
        y_pred = np.array(list(map(self.chouse_prob, y_proba)))
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
        y_predict = self.predict(X_batch)
        print(f"Log after predict X_batch shape {X_batch.shape} self.w.shape {self.w.shape}")
        ones = np.array(list(1 for i in range(y_predict.shape[0])))
        loss = np.mean(- (y_batch * np.log(y_predict) + (ones - y_batch) * np.log(ones - y_predict)))
        # Add regularization to the loss and gradient.
        # ругуляризация https://craftappmobile.com/l1-%D0%B8-l2-%D1%80%D0%B5%D0%B3%D1%83%D0%BB%D1%8F%D1%80%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D1%8F-%D0%B4%D0%BB%D1%8F-%D0%BB%D0%BE%D0%B3%D0%B8%D1%81%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%BE%D0%B9-%D1%80/?ysclid=l5cefgf0pn83923261

        print(f"X_batch shape {X_batch.shape} \n y_batch shape {y_batch.shape}")
        #dw = np.array([self.calculate_dw(X_batch, y_batch) for i in range(X_batch.shape[1])])
        # https://translated.turbopages.org/proxy_u/en-ru.ru.e4bc73ef-62c716fb-60995054-74722d776562/https/www.baeldung.com/cs/gradient-descent-logistic-regression
        # TODO убрать циклы
        print(f"Log self.w.shape {self.w.shape} dw {dw.shape}")
        print(f"Log len(self.w) {len(self.w)} dw {len(dw)}")
        # + 1 берется из-за того что длина вектора X[i] меньше чем длина вектора весов, возможно пересчитываются
        # только свободные члены, а w[0] не изменяется
        X_batch = LogisticRegression.append_biases(X_batch).toarray()
        for j in range(len(self.w[:-1])):
            for i in range(len(y_batch)):
                dw[j+1] += y_batch[i] - self.calculate_sigmoid(X_batch[i]) * X_batch[i][j]
            dw[j+1] = dw[j+1] / len(y_batch)  # Усреднение
            print(f"Log long cycle iteration j = {j} dw[j] = {dw[j]}")

        print(f"Log loss type dw {type(dw)}  {dw.shape}")

        # Note that you have to exclude bias term in regularization.


        return loss, dw

    @staticmethod
    def append_biases(X):
        return sparse.hstack((X, np.ones(X.shape[0])[:, np.newaxis])).tocsr()
