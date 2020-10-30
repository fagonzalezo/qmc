"""
Class for RBF Sampler with Orthogonal Random Features
"""
import warnings

import numpy as np
import scipy.stats as stats
from scipy.linalg import svd

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array, check_random_state, as_float_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics.pairwise import pairwise_kernels, KERNEL_PARAMS
from sklearn.utils.validation import check_non_negative, _deprecate_positional_args


class RBFSamplerORF(TransformerMixin, BaseEstimator):
    """Approximates feature map of an RBF kernel by Orthogonal Random Features
    of its Fourier transform.

    It implements a variant of Orthogonal Random Features.[1]

    Read more in the :ref:`User Guide <rbf_kernel_approx>`.

    Parameters
    ----------
    gamma : float
        Parameter of RBF kernel: exp(-gamma * x^2)

    n_components : int
        Number of Monte Carlo samples per original feature.
        Equals the dimensionality of the computed feature space.

    random_state : int, RandomState instance or None, optional (default=None)
        Pseudo-random number generator to control the generation of the random
        weights and random offset when fitting the training data.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    random_offset_ : ndarray of shape (n_components,), dtype=float64
        Random offset used to compute the projection in the `n_components`
        dimensions of the feature space.

    random_weights_ : ndarray of shape (n_features, n_components),\
        dtype=float64
        Random projection directions drawn from the Fourier transform
        of the RBF kernel.


    Examples
    --------
    >>> from sklearn.kernel_approximation import RBFSampler
    >>> from sklearn.linear_model import SGDClassifier
    >>> X = [[0, 0], [1, 1], [1, 0], [0, 1]]
    >>> y = [0, 0, 1, 1]
    >>> rbf_feature = RBFSamplerORF(gamma=1, random_state=1)
    >>> X_features = rbf_feature.fit_transform(X)
    >>> clf = SGDClassifier(max_iter=5, tol=1e-3)
    >>> clf.fit(X_features, y)
    SGDClassifier(max_iter=5)
    >>> clf.score(X_features, y)
    1.0

    Notes
    -----
    See "Orthogonal Random Features" by Felix, X et al.

    [1] "Orthogonal Random Features" by Felix, X et al.
    (https://arxiv.org/pdf/1610.09072)
    """
    @_deprecate_positional_args
    def __init__(self, *, gamma=1., n_components=100, random_state=None):
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the model with X.

        Samples random projection according to n_features.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the transformer.
        """

        #X = self._validate_data(X, accept_sparse='csr')
        random_state = check_random_state(self.random_state)
        n_features = X.shape[1]

        stack_random_weights = []
        for i in range(round(self.n_components / n_features)+1):
            random_gaussian_weights_ = random_state.normal(size=(n_features, n_features))
            q, _ = np.linalg.qr(random_gaussian_weights_, mode='reduced')
            random_chi_weights = random_state.chisquare(df=n_features, size=(n_features))
            random_chi_weights = np.sqrt(random_chi_weights)

            random_chi_weights = np.diag(random_chi_weights)

            random_weights_ = np.dot(random_chi_weights, q)
            stack_random_weights.append(random_weights_)

        self.random_weights_ = np.sqrt(2 * self.gamma) * np.hstack(stack_random_weights)[:n_features, :self.n_components]
        self.random_offset_ = random_state.uniform(0, 2 * np.pi, size=self.n_components)

        return self

    def transform(self, X):
        """Apply the approximate feature map to X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        check_is_fitted(self)

        X = check_array(X, accept_sparse='csr')
        projection = safe_sparse_dot(X, self.random_weights_)
        projection += self.random_offset_
        np.cos(projection, projection)
        projection *= np.sqrt(2.) / np.sqrt(self.n_components)
        return projection