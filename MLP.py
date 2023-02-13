from copy import copy
import numpy as np
from scipy.special import expit
import sys

from itertools import chain

# Implémentation du MLP
# Couche d'entrées, 1 couche cachée et couche de sorties
#
# Code similaire à Adaline

class NeuralNetMLP(object):
    """ Feedforward neural network / Multi-layer perceptron classifier.

    Parameters
    ------------
    n_output : int
        Number of output units, should be equal to the number of unique class labels.
    n_features : int
        Number of features (dimensions) in the target dataset.Should be equal to the number of columns in the X array.
    n_hidden : int (default: 30)
        Number of hidden units.
    l1 : float (default: 0.0)
        Lambda value for L1-regularization. No regularization if l1=0.0 (default)
    l2 : float (default: 0.0)
        Lambda value for L2-regularization. No regularization if l2=0.0 (default)
    epochs : int (default: 500)
        Number of passes over the training set.
    eta : float (default: 0.001)
        Learning rate.
    alpha : float (default: 0.0)
        Momentum constant. Factor multiplied with the gradient of the previous epoch t-1 to improve learning speed
        w(t) := w(t) - (grad(t) + alpha*grad(t-1))
    decrease_const : float (default: 0.0)
        Decrease constant. Shrinks the learning rate after each epoch via eta / (1 + epoch*decrease_const)
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent circles.
    minibatches : int (default: 1)
        Divides training data into k minibatches for efficiency. Normal gradient descent learning if k=1 (default).
    random_state : int (default: None)
        Set random state for shuffling and initializing the weights.

    Attributes
    -----------
    cost_ : list
      Sum of squared errors after each epoch.

    """

    def __init__(self, n_output, n_features, n_hidden=[30, ], l1=0.0, l2=0.0, epochs=500, eta=0.001, alpha=0.0,
                 decrease_const=0.0, shuffle=True, minibatches=1, random_state=None):

        np.random.seed(random_state)
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        # self.w1, self.w2 = self._initialize_weights()
        self.weights = self._initialize_weights()

        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.decrease_const = decrease_const
        self.shuffle = shuffle
        self.minibatches = minibatches

    def _encode_labels(self, y, k):
        """Encode labels into one-hot representation

        Parameters
        ------------
        y : array, shape = [n_samples]   Target values.

        Returns
        -----------
        onehot : array, shape = (n_labels, n_samples)

        """
        onehot = np.zeros((k, y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.0
        return onehot

    def _initialize_weights(self):
        """Initialize weights with small random numbers."""
        weights = list()
        sizes = list(chain([self.n_features], self.n_hidden, [self.n_output]))

        for s1, s2 in zip(sizes[:-1], sizes[1:]):
            w = np.random.uniform(-1, 1, size=(s2, s1+1))
            weights.append(w)
        # for n_hidden in self.n_hidden:
        #     w1 = np.random.uniform(-1.0, 1.0,
        #                         size=n_hidden * (self.n_features + 1))
        #     w1 = w1.reshape(n_hidden, self.n_features + 1)
        #     weights.append(w1)

        # w2 = np.random.uniform(-1.0, 1.0,
        #                        size=self.n_output * (n_hidden + 1))
        # w2 = w2.reshape(self.n_output, n_hidden + 1)
        # weights.append(w2)

        return weights

    def _sigmoid(self, z):
        """Compute logistic function (sigmoid)

        Uses scipy.special.expit to avoid overflow
        error for very small input values z.

        """
        # return 1.0 / (1.0 + np.exp(-z))
        return expit(z)

    def _sigmoid_gradient(self, z):
        """Compute gradient of the logistic function"""
        sg = self._sigmoid(z)
        return sg * (1.0 - sg)

    def _add_bias_unit(self, X, how='column'):
        """Add bias unit (column or row of 1s) to array at index 0"""
        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0] + 1, X.shape[1]))
            X_new[1:, :] = X
        else:
            raise AttributeError('`how` must be `column` or `row`')
        return X_new

    def _del_bias_unit(self, X, how='column'):
        """Del bias unit (column or row of 1s) to array at index 0"""
        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1] - 1))
            X_new = X[:, 1:]
        elif how == 'row':
            X_new = np.ones((X.shape[0] + 1, X.shape[1]))
            X_new = X[1:, :]
        else:
            raise AttributeError('`how` must be `column` or `row`')
        return X_new        

    def _feedforward(self, X, weights):
        """Compute feedforward step

        Parameters
        -----------
        X : array, shape = [n_samples, n_features]
            Input layer with original features.

        weights: an Iterable of weights
        # w1 : array, shape = [n_hidden_units, n_features]
        #     Weight matrix for input layer -> hidden layer.
        # w2 : array, shape = [n_output_units, n_hidden_units]
        #     Weight matrix for hidden layer -> output layer.

        Returns
        ----------
        ax: iterable of a1, a2, ..., an
        zx: iterable of z2, z3, ..., zn
        # a1 : array, shape = [n_samples, n_features+1]
        #     Input values with bias unit.
        # a2 : array, shape = [n_hidden+1, n_samples]
        #     Activation of hidden layer.
        # a3 : array, shape = [n_output_units, n_samples]
        #     Activation of output layer.
        # z2 : array, shape = [n_hidden, n_samples]
        #     Net input of hidden layer.
        # z3 : array, shape = [n_output_units, n_samples]
        #     Net input of output layer.

        """
        a1 = self._add_bias_unit(X, how='column')
        ax = list()
        zx = list()
        ax.append(a1)

        a_temp = a1.T
        for w in weights:
            z_temp = w.dot(a_temp)
            a_temp = self._sigmoid(z_temp)
            a_temp = self._add_bias_unit(a_temp, how='row')
            ax.append(a_temp)
            zx.append(z_temp)

        # z2 = w1.dot(a1.T)
        # a2 = self._sigmoid(z2)
        # a2 = self._add_bias_unit(a2, how='row')
        # z3 = w2.dot(a2)
        # a3 = self._sigmoid(z3)
        # return a1, z2, a2, z3, a3
        a_end = ax.pop()
        a_end = self._del_bias_unit(a_end, how='row')
        ax.append(a_end)
        return ax, zx

    # old version
    # def _L2_reg(self, lambda_, w1, w2):
    #     """Compute L2-regularization cost"""
    #     return (lambda_ / 2.0) * (np.sum(w1[:, 1:] ** 2) +
    #                               np.sum(w2[:, 1:] ** 2))
    # def _L1_reg(self, lambda_, w1, w2):
    #     """Compute L1-regularization cost"""
    #     return (lambda_ / 2.0) * (np.abs(w1[:, 1:]).sum() +
    #                               np.abs(w2[:, 1:]).sum())

    def _L2_reg(self, lambda_, weights):
        """Compute L2-regularization cost"""
        values = [np.sum(w[:, 1:]) ** 2 for w in weights]
        return (lambda_ / 2.0) * sum(values)


    def _L1_reg(self, lambda_, weights):
        """Compute L1-regularization cost"""
        values = [np.sum(np.abs((w[:, 1:]))) for w in weights]
        return (lambda_ / 2.0) * sum(values)

    def _get_cost(self, y_enc, output, weights):
        # TODO： change definition here
        """Compute cost function.

        Parameters
        ----------
        y_enc : array, shape = (n_labels, n_samples)
            one-hot encoded class labels.
        output : array, shape = [n_output_units, n_samples]
            Activation of the output layer (feedforward)

        # the follwing defnition is no longer appliable
        # w1 : array, shape = [n_hidden_units, n_features]
        #     Weight matrix for input layer -> hidden layer.
        # w2 : array, shape = [n_output_units, n_hidden_units]
        #     Weight matrix for hidden layer -> output layer.

        Returns
        ---------
        cost : float
            Regularized cost.

        """
        term1 = -y_enc * (np.log(output))
        term2 = (1.0 - y_enc) * np.log(1.0 - output)
        cost = np.sum(term1 - term2)

        L1_term = self._L1_reg(self.l1, weights)
        L2_term = self._L2_reg(self.l2, weights)
        cost = cost + L1_term + L2_term
        return cost

    #
    # Nous verrons plus tard
    #

    # TODO: change definition here
    # def _get_gradient(self, a1, a2, a3, z2, y_enc, w1, w2):
    def _get_gradient(self, ax, zx, y_enc, weights):
        """ Compute gradient step using backpropagation.

        Parameters
        ------------
        a1 : array, shape = [n_samples, n_features+1]
            Input values with bias unit.
        a2 : array, shape = [n_hidden+1, n_samples]
            Activation of hidden layer.
        a3 : array, shape = [n_output_units, n_samples]
            Activation of output layer.
        z2 : array, shape = [n_hidden, n_samples]
            Net input of hidden layer.
        y_enc : array, shape = (n_labels, n_samples)
            one-hot encoded class labels.
        w1 : array, shape = [n_hidden_units, n_features]
            Weight matrix for input layer -> hidden layer.
        w2 : array, shape = [n_output_units, n_hidden_units]
            Weight matrix for hidden layer -> output layer.

        Returns
        ---------
        grad1 : array, shape = [n_hidden_units, n_features]
            Gradient of the weight matrix w1.
        grad2 : array, shape = [n_output_units, n_hidden_units]
            Gradient of the weight matrix w2.

        """
        # backpropagation
        # sigma3 = a3 - y_enc  # erreur de classification
        # z2 = self._add_bias_unit(z2, how='row')
        # sigma2 = w2.T.dot(sigma3) * self._sigmoid_gradient(z2)
        # sigma2 = sigma2[1:, :]
        # grad1 = sigma2.dot(a1)
        # grad2 = sigma3.dot(a2.T)

        # calculate the last sigma value
        sigmas = list()
        sigmas.append(ax[-1] - y_enc)
        sigma = sigmas[-1]

        # exclude the last z value, which is not used in the calculation; besides, this line makes a copy
        zx = zx[:-1]
        
        # make a copy of weights since it will be used later
        weights_2 = copy(weights)

        # for similar reason, treat the weights values
        weights = weights[1:]

        # transpose the first value of ax, make a copy
        ax = [ax[0].T] + ax [1:]
        
        for z, w in zip(reversed(zx), reversed(weights)):
            z = self._add_bias_unit(z, how='row')
            sigma = w.T.dot(sigma) * self._sigmoid_gradient(z)
            sigma = sigma[1:, :]
            sigmas.append(sigma)

        # reverse (back) the result of sigma for the next step
        sigmas.reverse()

        grads = list()
        for a, sigma, w in zip(ax, sigmas, weights_2):
            temp = sigma.dot(a.T)
            temp[:, 1:] += self.l2 * w[:, 1:]
            temp[:, 1:] += self.l1 * np.sign(w[:, 1:])
            grads.append(temp)


        # # regularize
        # grad1[:, 1:] += self.l2 * w1[:, 1:]
        # grad1[:, 1:] += self.l1 * np.sign(w1[:, 1:])
        # grad2[:, 1:] += self.l2 * w2[:, 1:]
        # grad2[:, 1:] += self.l1 * np.sign(w2[:, 1:])
        # return grad1, grad2

        return grads

    def predict(self, X):
        """Predict class labels

        Parameters
        -----------
        X : array, shape = [n_samples, n_features]
            Input layer with original features.

        Returns:
        ----------
        y_pred : array, shape = [n_samples]
            Predicted class labels.

        """
        if len(X.shape) != 2:
            raise AttributeError('X must be a [n_samples, n_features] array.\n'
                                 'Use X[:,None] for 1-feature classification,'
                                 '\nor X[[i]] for 1-sample classification')

        ax, zx = self._feedforward(X, self.weights)
        y_pred = np.argmax(zx[-1], axis=0)
        return y_pred

    #
    # Fonction d'entraînement
    #
    def fit(self, X, y, print_progress=False):
        """ Learn weights from training data.

        Parameters
        -----------
        X : array, shape = [n_samples, n_features]
            Input layer with original features.
        y : array, shape = [n_samples]
            Target class labels.
        print_progress : bool (default: False)
            Prints progress as the number of epochs
            to stderr.

        Returns:
        ----------
        self

        """
        self.cost_ = []
        X_data, y_data = X.copy(), y.copy()
        y_enc = self._encode_labels(y, self.n_output)  # Vecteur one-hot

        # delta_w1_prev = np.zeros(self.w1.shape)
        # delta_w2_prev = np.zeros(self.w2.shape)
        # the following line replace the two previous line
        delta_weights_prev = [np.zeros_like(w) for w in self.weights]

        for i in range(self.epochs):  # Nombre de passage sur le dataset

            # adaptive learning rate
            self.eta /= (1 + self.decrease_const * i)  # Permet de réduire le nombre d'epochs nécessaire à la convergence en limitant les risques de "pas" trop grand!

            if print_progress:
                sys.stderr.write('\rEpoch: %d/%d' % (i + 1, self.epochs))
                sys.stderr.flush()

            if self.shuffle:  # on mélange le dataset à chaque epoch
                idx = np.random.permutation(y_data.shape[0])
                X_data, y_enc = X_data[idx], y_enc[:, idx]

            mini = np.array_split(range(y_data.shape[0]),
                                  self.minibatches)  # Si le mode minibatch est activé, le dataset en entrée est divisé en batch pour le calcul des gradients
            for idx in mini:
                # feedforward

                ax, zx = self._feedforward(X_data[idx], self.weights) 

                cost = self._get_cost(y_enc=y_enc[:, idx], output=ax[-1], weights=self.weights)
                self.cost_.append(cost)

                # compute gradient via backpropagation
                #
                # Nous verrons plus en détails
                # grad1, grad2 = self._get_gradient(a1=a1, a2=a2, a3=a3, z2=z2, y_enc=y_enc[:, idx], w1=self.w1, w2=self.w2)
                # delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2

                # TODO: change syntax here
                grads = self._get_gradient(ax = ax, zx=zx, y_enc=y_enc[:, idx], weights=self.weights)
                deltas = [self.eta * g for g in grads]


                # self.w1 -= (delta_w1 + (self.alpha * delta_w1_prev))
                # self.w2 -= (delta_w2 + (self.alpha * delta_w2_prev))
                # delta_w1_prev, delta_w2_prev = delta_w1, delta_w2

                # the following two lines, alone with previous changes, replace the three lines above (commented)
                self.weights = [weight - delta + (self.alpha * delta_prev) for weight, delta, delta_prev in zip(self.weights, deltas, delta_weights_prev)]
                delta_weights_prev = deltas

        return self

# Retour sur le powerpoint
