import numpy as np
from sklearn.utils import check_scalar, check_array, check_random_state
from sklearn.decomposition import FastICA

import os
import shutil
import subprocess
import tempfile


class LiNGD:
    """Implementation of LiNG Discovery algorithm [1]_

    References
    ----------
    .. [1] Gustavo Lacerda, Peter Spirtes, Joseph Ramsey, and Patrik O. Hoyer.
       Discovering cyclic causal models by independent components analysis.
       In Proceedings of the Twenty-Fourth Conference on Uncertainty
       in Artificial Intelligence (UAI'08). AUAI Press, Arlington, Virginia, USA, 366–374.
    """

    def __init__(self, k=5, random_state=None):
        """ Construct a LiNG-D model.

        Parameters
        ----------
        k : int, option (default=5)
            Number of candidate causal graphs to estimate.
        """
        k = check_scalar(k, "k", int, min_val=1)
        random_state = check_random_state(random_state)

        self._k = k
        self._random_state = random_state

        self._fitted = False

    def fit(self, X):
        """ Fit the model to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = check_array(X)

        ica = FastICA(random_state=self._random_state)
        ica.fit_transform(X)
        W_ica = ica.components_

        permutes, costs = self._run_murty(1 / np.abs(W_ica), k=self._k)
        costs = np.array(costs)

        estimated_Bs = []
        for i, p in enumerate(permutes):
            PW_ica = np.zeros_like(W_ica)
            PW_ica[p] = W_ica

            D = np.diag(PW_ica)[:, np.newaxis]

            W_estimate = PW_ica / D
            B_estimate = np.eye(len(W_estimate)) - W_estimate

            estimated_Bs.append(B_estimate)
        estimated_Bs = np.array(estimated_Bs)

        is_stables = []
        for B in estimated_Bs:
            values, _ = np.linalg.eig(B)
            is_stables.append(all(abs(values) < 1))
        is_stables = np.array(is_stables)

        self._X = X
        self._adjacency_matrices = estimated_Bs
        self._costs = costs
        self._is_stables = is_stables
        self._fitted = True

        return self

    def bound_of_causal_effect(self, target_index):
        """
        This method calculates the causal effect from target_index to each feature

        Parameters
        ----------
        target_index : int
            The index of the intervention target.

        Returns
        -------
        causal_effects : array-like, shape (k, n_features)
            list of causal effects.
        """

        self._check_is_fitted()

        target_index = check_scalar(
            target_index,
            "target_index",
            int,
            min_val=0,
            max_val=self._X.shape[1] - 1
        )

        aces = []
        for B in self._adjacency_matrices:
            X1 = self._intervention(self._X, B, target_index, 1)
            X0 = self._intervention(self._X, B, target_index, 0)
            ace = (X1.mean(axis=0) - X0.mean(axis=0)).tolist()
            aces.append(ace)

        return np.array(aces)

    @property
    def adjacency_matrices_(self):
        self._check_is_fitted()
        return self._adjacency_matrices

    @property
    def costs_(self):
        self._check_is_fitted()
        return self._costs

    @property
    def is_stables_(self):
        self._check_is_fitted()
        return self._is_stables

    def _run_murty(self, X, k):
        # XXX: muRty occurs an error if X has 2 variables and k is greater than 2.
        if X.shape[1] == 2:
            k = 1

        try:
            temp_dir = tempfile.mkdtemp()

            path = os.path.join(os.path.dirname(__file__), "murty.r")

            args = [f"--temp_dir={temp_dir}"]

            np.savetxt(os.path.join(temp_dir, "X.csv"), X, delimiter=",")
            np.savetxt(os.path.join(temp_dir, "k.csv"), [k], delimiter=",")

            # run
            ret = subprocess.run(["Rscript", path, *args], capture_output=True)
            if ret.returncode != 0:
                if ret.returncode == 2:
                    msg = "muRty is not installed."
                else:
                    msg = ret.stderr.decode()
                raise RuntimeError(msg)

            # retrieve result
            permutes = []

            for f in os.listdir(temp_dir):
                if not f.startswith("solution"):
                    continue

                solution = np.loadtxt(os.path.join(temp_dir, f), delimiter=",", skiprows=1)

                permute = [x[1] for x in np.argwhere(solution > 0)]
                permutes.append(permute)

            costs = np.loadtxt(os.path.join(temp_dir, "costs.csv"), delimiter=",", skiprows=1)
            costs = np.array(costs).flatten().tolist()
        except FileNotFoundError:
            raise RuntimeError("Rscript is not found.")
        except BaseException as e:
            raise RuntimeError(str(e))
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

        return permutes, costs

    def _check_is_fitted(self):
        if not self._fitted:
            raise RuntimeError("This instance is not fitted yet. Call 'fit' with \
                appropriate arguments before using this instance.")

    def _intervention(self, X, B, target_index, value):
        # estimate error terms
        e = ((np.eye(len(B)) - B) @ X.T).T

        # set the given intervention value
        e[:, target_index] = value

        # resample errors
        resample_index = np.random.choice(np.arange(len(e)), size=len(e))
        e = e[resample_index]

        # remove edges
        B_ = B.copy()
        B_[target_index, :] = 0

        # generate data
        A = np.linalg.inv(np.eye(len(B)) - B_)
        X_ = (A @ e.T).T

        return X_
