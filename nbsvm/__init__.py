import numpy as np

from scipy.sparse import spmatrix, coo_matrix

from sklearn.base import BaseEstimator
from sklearn.linear_model.base import LinearClassifierMixin, SparseCoefMixin
from sklearn.svm import LinearSVC


__all__ = ['NBSVM']

class NBSVM(BaseEstimator, LinearClassifierMixin, SparseCoefMixin):

    def __init__(self, alpha=1, C=1, beta=0.25):
        self.alpha = alpha
        self.C = C
        self.beta = beta

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) == 2:
            self.coef_ = self._fit_binary(X, y)
            self.intercept_ = 0
        else:
            self.coef_ = np.concatenate([
                self._fit_binary(X, y == class_)
                for class_ in self.classes_
            ])
            self.intercept_ = np.zeros(len(self.coef_))
        return self

    def _fit_binary(self, X, y):
        p = np.asarray(self.alpha + X[y == 1].sum(axis=0))[0]
        q = np.asarray(self.alpha + X[y == 0].sum(axis=0))[0]
        r = np.log(p/np.abs(p).sum()) - np.log(q/np.abs(q).sum())

        if isinstance(X, spmatrix):
            indices = np.arange(len(r))
            r_sparse = coo_matrix(
                (r, (indices, indices)),
                shape=(len(r), len(r))
            )
            X_scaled = X * r_sparse
        else:
            X_scaled = X * r

        lsvc = LinearSVC(
            C=self.C,
            fit_intercept=False,
            max_iter=10000
        ).fit(X_scaled, y)

        return r * (
            (1 - self.beta) * np.abs(lsvc.coef_).mean() +
            self.beta * lsvc.coef_
        )
