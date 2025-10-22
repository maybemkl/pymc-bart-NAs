# pylint: disable=unused-argument
# pylint: disable=arguments-differ
#   Copyright 2022 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import warnings
from multiprocessing import Manager
from typing import Optional, Literal

import numpy as np
import numpy.typing as npt
import pytensor.tensor as pt
from pandas import DataFrame, Series
from pymc.distributions.distribution import Distribution, _support_point
from pymc.logprob.abstract import _logprob
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.sharedvar import TensorSharedVariable
from pytensor.tensor.variable import TensorVariable

from .split_rules import (
    SplitRule,
    ContinuousSplitRule,
    MissingnessAwareSplitRule,
    MissingnessAwareCategoricalSplitRule,
)
from .utils import TensorLike, _sample_posterior

# Global registry for storing BART trees across BART Op replacements
_bart_trees_registry = {}

__all__ = ["BART"]


class BARTRV(RandomVariable):
    """Base class for BART."""

    name: str = "BART"
    signature = "(m,n),(m),(),(),() -> (m)"
    dtype: str = "floatX"
    _print_name: tuple[str, str] = ("BART", "\\operatorname{BART}")

    def _supp_shape_from_params(self, dist_params, rep_param_idx=1, param_shapes=None):  # pylint: disable=arguments-renamed
        idx = dist_params[0].ndim - 2
        return [dist_params[0].shape[idx]]

    @classmethod
    def rng_fn(  # pylint: disable=W0237
        cls, rng=None, X=None, Y=None, m=None, alpha=None, beta=None, size=None
    ):
        if not size:
            size = None

        # Try to get trees from BART Op first
        all_trees = []
        if hasattr(cls, 'all_trees') and cls.all_trees:
            all_trees = cls.all_trees
            print(f"DEBUG: rng_fn - found trees in BART Op, length: {len(all_trees)}")
        else:
            # If no trees in BART Op, try to load from pickle file
            trees_key = f"BART_trees"
            pickle_file = f"bart_trees_{trees_key}.pkl"
            try:
                import cloudpickle as cpkl
                with open(pickle_file, 'rb') as f:
                    loaded_trees = cpkl.load(f)
                print(f"DEBUG: rng_fn - loaded from file, type: {type(loaded_trees)}, length: {len(loaded_trees)}")
                # loaded_trees is now a list of tree sets (one per output dimension)
                # _sample_posterior expects a list of tree sets, so wrap it
                all_trees = [loaded_trees]
                print(f"DEBUG: rng_fn - using all_trees length: {len(all_trees)}")
            except Exception as e:
                print(f"DEBUG: rng_fn - failed to load from file: {e}")
                # During model initialization, return mean if no trees available
                print(f"DEBUG: rng_fn - returning mean (no trees available)")
        
        if len(all_trees) == 0:
            if isinstance(cls.Y, (TensorSharedVariable, TensorVariable)):
                Y = cls.Y.eval()
            else:
                Y = cls.Y

            if size is not None:
                return np.full((size[0], Y.shape[0]), Y.mean())
            else:
                return np.full(Y.shape[0], Y.mean())
        else:
            if size is not None:
                shape = size[0]
            else:
                shape = 1
            
            # Use current X value (from shared variable if applicable) for prediction
            # This ensures predictions use updated data from set_value() calls
            if isinstance(cls.X, (TensorSharedVariable, TensorVariable)):
                current_X = cls.X.eval()
            else:
                current_X = cls.X
            
            result = _sample_posterior(all_trees, current_X, rng=rng, shape=shape).squeeze().T
            return result


bart = BARTRV()


class BART(Distribution):
    r"""
    Bayesian Additive Regression Tree distribution.

    Distribution representing a sum over trees

    Parameters
    ----------
    X : PyTensor Variable, Pandas/Polars DataFrame or Numpy array
        The covariate matrix.
    Y : PyTensor Variable, Pandas/Polar DataFrame/Series,or Numpy array
        The response vector.
    m : int
        Number of trees.
    response : str
        How the leaf_node values are computed. Available options are ``constant``, ``linear`` or
        ``mix``. Defaults to ``constant``. Options ``linear`` and ``mix`` are still experimental.
    alpha : float
        Controls the prior probability over the depth of the trees.
        Should be in the (0, 1) interval.
    beta : float
        Controls the prior probability over the number of leaves of the trees.
        Should be positive.
    split_prior : Optional[list[float]], default None.
        List of positive numbers, one per column in input data.
        Defaults to None, all covariates have the same prior probability to be selected.
    split_rules : Optional[list[SplitRule]], default None
        List of SplitRule objects, one per column in input data.
        Allows using different split rules for different columns. Default is ContinuousSplitRule.
        Other options are OneHotSplitRule and SubsetSplitRule, both meant for categorical variables.
    separate_trees : Optional[bool], default False
        When training multiple trees (by setting a shape parameter), the default behavior is to
        learn a joint tree structure and only have different leaf values for each.
        This flag forces a fully separate tree structure to be trained instead.
        This is unnecessary in many cases and is considerably slower, multiplying
        run-time roughly by number of dimensions.
    missingness_handling : Literal["filter", "enhanced", "aware"], default "enhanced"
        [NA-handling] Strategy for handling missing values in the data:
        
        - "filter": Legacy behavior - missing values are filtered out during splits
        - "enhanced": Enhanced standard split rules that handle missing values gracefully
        - "aware": Missingness-aware splits that treat missing values as a separate category
                  (similar to bartMachine's approach)
    auto_detect_categorical : bool, default True
        [NA-handling] Automatically detect categorical variables and use appropriate split rules.
        When True, variables with fewer than 10 unique values are treated as categorical.

    Notes
    -----
    The parameters ``alpha`` and ``beta`` parametrize the probability that a node at
    depth :math:`d \: (= 0, 1, 2,...)` is non-terminal, given by :math:`\alpha(1 + d)^{-\beta}`.
    The default values are :math:`\alpha = 0.95` and :math:`\beta = 2`.

    This is the recommend prior by Chipman Et al. BART: Bayesian additive regression trees,
    `link <https://doi.org/10.1214/09-AOAS285>`__
    
    Missingness Handling
    --------------------
    [NA-handling] The package now supports sophisticated missing value handling:
    
    1. **Enhanced Standard Rules**: Existing split rules have been enhanced to handle missing values
       gracefully without filtering them out.
    
    2. **Missingness-Aware Splits**: New split rules that explicitly treat missing values as a
       separate category, allowing the model to learn patterns in missingness itself.
    
    3. **Automatic Categorical Detection**: The model can automatically detect categorical variables
       and apply appropriate missingness handling strategies.
    """

    def __new__(
        cls,
        name: str,
        X: TensorLike,
        Y: TensorLike,
        m: int = 50,
        alpha: float = 0.95,
        beta: float = 2.0,
        response: str = "constant",
        split_prior: Optional[npt.NDArray] = None,
        split_rules: Optional[list[SplitRule]] = None,
        separate_trees: Optional[bool] = False,
        missingness_handling: Literal["filter", "enhanced", "aware"] = "enhanced",
        auto_detect_categorical: bool = True,
        **kwargs,
    ):
        if response in ["linear", "mix"]:
            warnings.warn(
                "Options linear and mix are experimental and still not well tested\n"
                + "Use with caution."
            )
        
        # [NA-handling] Validate missingness handling parameter
        if missingness_handling not in ["filter", "enhanced", "aware"]:
            raise ValueError(
                f"missingness_handling must be one of ['filter', 'enhanced', 'aware'], "
                f"got {missingness_handling}"
            )
        
        # Create a regular list for each BART instance (no multiprocessing)
        instance_all_trees = []

        X, Y = preprocess_xy(X, Y)

        split_prior = np.array([]) if split_prior is None else np.asarray(split_prior)

        # [NA-handling] Auto-detect categorical variables and set up split rules
        if split_rules is None and auto_detect_categorical:
            split_rules = cls._auto_detect_split_rules(X, missingness_handling)
        elif split_rules is None:
            # [NA-handling] Use enhanced split rules by default
            split_rules = cls._get_default_split_rules(X.shape[1], missingness_handling)

        bart_op = type(
            f"BART_{name}",
            (BARTRV,),
            {
                "name": "BART",
                "all_trees": [],  # Use empty list initially
                "inplace": False,
                "initval": Y.mean(),
                "X": X,
                "Y": Y,
                "m": m,
                "response": response,
                "alpha": alpha,
                "beta": beta,
                "split_prior": split_prior,
                "split_rules": split_rules,
                "separate_trees": separate_trees,
                "missingness_handling": missingness_handling,  # [NA-handling] Store missingness handling strategy
            },
        )()
        
        # Set the all_trees reference after creation
        bart_op.all_trees = instance_all_trees

        Distribution.register(BARTRV)

        @_support_point.register(BARTRV)
        def get_moment(rv, size, *rv_inputs):
            return cls.get_moment(rv, size, *rv_inputs)

        cls.rv_op = bart_op
        params = [X, Y, m, alpha, beta]
        return super().__new__(cls, name, *params, **kwargs)

    @classmethod
    def _auto_detect_split_rules(cls, X: npt.NDArray, missingness_handling: str) -> list[SplitRule]:
        """
        [NA-handling] Automatically detect categorical variables and create appropriate split rules.
        
        Parameters:
        -----------
        X : npt.NDArray
            Input data matrix
        missingness_handling : str
            Missingness handling strategy
            
        Returns:
        --------
        list[SplitRule]
            List of split rules, one per column
        """
        split_rules = []
        
        for col in range(X.shape[1]):
            unique_values = np.unique(X[:, col])
            # Filter out NaN values for detection
            unique_values = unique_values[~np.isnan(unique_values)]
            
            # Consider variable categorical if it has fewer than 10 unique values
            if len(unique_values) < 10:
                if missingness_handling == "aware":
                    split_rules.append(MissingnessAwareCategoricalSplitRule())
                else:
                    # Use enhanced subset split rule for categorical variables
                    from .split_rules import SubsetSplitRule
                    split_rules.append(SubsetSplitRule())
            else:
                if missingness_handling == "aware":
                    split_rules.append(MissingnessAwareSplitRule())
                else:
                    # Use enhanced continuous split rule for continuous variables
                    split_rules.append(ContinuousSplitRule())
        
        return split_rules

    @classmethod
    def _get_default_split_rules(cls, n_columns: int, missingness_handling: str) -> list[SplitRule]:
        """
        [NA-handling] Get default split rules based on missingness handling strategy.
        
        Parameters:
        -----------
        n_columns : int
            Number of columns in the data
        missingness_handling : str
            Missingness handling strategy
            
        Returns:
        --------
        list[SplitRule]
            List of default split rules
        """
        if missingness_handling == "aware":
            return [MissingnessAwareSplitRule() for _ in range(n_columns)]
        else:
            return [ContinuousSplitRule() for _ in range(n_columns)]

    @classmethod
    def dist(cls, *params, **kwargs):
        # Be compatible with PyMC versions that expect either unpacked or packed params
        try:
            return super().dist(*params, **kwargs)
        except TypeError:
            return super().dist(params, **kwargs)

    def logp(self, x, *inputs):
        """Calculate log probability.

        Parameters
        ----------
        x: numeric, TensorVariable
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """
        return pt.zeros_like(x)

    @classmethod
    def get_moment(cls, rv, size, *rv_inputs):
        mean = pt.fill(size, rv.Y.mean())
        return mean


def preprocess_xy(X: TensorLike, Y: TensorLike) -> tuple[npt.NDArray, npt.NDArray]:
    """
    [NA-handling] Enhanced preprocessing that preserves missing values for missingness-aware handling.
    """
    if isinstance(Y, (Series, DataFrame)):
        Y = Y.to_numpy()
    if isinstance(X, (Series, DataFrame)):
        X = X.to_numpy()

    try:
        import polars as pl

        if isinstance(X, (pl.Series, pl.DataFrame)):
            X = X.to_numpy()
        if isinstance(Y, (pl.Series, pl.DataFrame)):
            Y = Y.to_numpy()
    except ImportError:
        pass

    Y = Y.astype(float)
    X = X.astype(float)

    return X, Y


@_logprob.register(BARTRV)
def logp(op, value_var, *dist_params, **kwargs):
    _dist_params = dist_params[3:]
    value_var = value_var[0]
    return BART.logp(value_var, *_dist_params)  # pylint: disable=no-value-for-parameter
