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

from abc import abstractmethod

import numpy as np
from numba import njit


class SplitRule:
    """
    Abstract template class for a split rule
    """

    @staticmethod
    @abstractmethod
    def get_split_value(available_splitting_values):
        pass

    @staticmethod
    @abstractmethod
    def divide(available_splitting_values, split_value):
        pass


class ContinuousSplitRule(SplitRule):
    """
    Standard continuous split rule: pick a pivot value and split
    depending on if variable is smaller or greater than the value picked.
    
    [NA-handling] Enhanced to handle missing values by treating them as a separate category
    that can be routed to either the left or right child based on the split decision.
    """

    @staticmethod
    def get_split_value(available_splitting_values):
        split_value = None
        if available_splitting_values.size > 1:
            # [NA-handling] Filter out NaN values for split value selection
            non_nan_values = available_splitting_values[~np.isnan(available_splitting_values)]
            if non_nan_values.size > 1:
                idx_selected_splitting_values = int(
                    np.random.random() * len(non_nan_values)
                )
                split_value = non_nan_values[idx_selected_splitting_values]
        return split_value

    @staticmethod
    @njit
    def divide(available_splitting_values, split_value):
        # [NA-handling] Handle missing values by routing them to the left child
        # This is a simple default strategy; more sophisticated approaches can be implemented
        result = available_splitting_values <= split_value
        # NaN values get routed to the left child (True)
        nan_mask = np.isnan(available_splitting_values)
        result[nan_mask] = True
        return result


class OneHotSplitRule(SplitRule):
    """
    Choose a single categorical value and branch on if the variable is that value or not
    
    [NA-handling] Enhanced to handle missing values by treating them as a separate category
    that can be routed to either the left or right child based on the split decision.
    """

    @staticmethod
    def get_split_value(available_splitting_values):
        split_value = None
        # [NA-handling] Filter out NaN values for split value selection
        non_nan_values = available_splitting_values[~np.isnan(available_splitting_values)]
        if non_nan_values.size > 1 and not np.all(
            non_nan_values == non_nan_values[0]
        ):
            idx_selected_splitting_values = int(
                np.random.random() * len(non_nan_values)
            )
            split_value = non_nan_values[idx_selected_splitting_values]
        return split_value

    @staticmethod
    @njit
    def divide(available_splitting_values, split_value):
        # [NA-handling] Handle missing values by routing them to the left child
        # This treats missing values as a separate category that goes to the left
        result = available_splitting_values == split_value
        # NaN values get routed to the left child (True)
        nan_mask = np.isnan(available_splitting_values)
        result[nan_mask] = True
        return result


class SubsetSplitRule(SplitRule):
    """
    Choose a random subset of the categorical values and branch on belonging to that set.
    This is the approach taken by Sameer K. Deshpande.
    flexBART: Flexible Bayesian regression trees with categorical predictors. arXiv,
    `link <https://arxiv.org/abs/2211.04459>`__
    
    [NA-handling] Enhanced to handle missing values by treating them as a separate category
    that can be included in the subset or not based on the split decision.
    """

    @staticmethod
    def get_split_value(available_splitting_values):
        split_value = None
        # [NA-handling] Filter out NaN values for split value selection
        non_nan_values = available_splitting_values[~np.isnan(available_splitting_values)]
        if non_nan_values.size > 1 and not np.all(
            non_nan_values == non_nan_values[0]
        ):
            unique_values = np.unique(non_nan_values)
            while True:
                sample = np.random.randint(0, 2, size=len(unique_values)).astype(bool)
                if np.any(sample):
                    break
            split_value = unique_values[sample]
        return split_value

    @staticmethod
    def divide(available_splitting_values, split_value):
        # [NA-handling] Handle missing values by including them in the subset with 50% probability
        # This treats missing values as a separate category that can be included or excluded
        result = np.isin(available_splitting_values, split_value)
        # NaN values get included in the subset with 50% probability
        nan_mask = np.isnan(available_splitting_values)
        nan_indices = np.where(nan_mask)[0]
        if len(nan_indices) > 0:
            # Randomly assign NaN values to the subset (True) or not (False)
            nan_assignments = np.random.choice([True, False], size=len(nan_indices))
            result[nan_indices] = nan_assignments
        return result


class MissingnessAwareSplitRule(SplitRule):
    """
    [NA-handling] NEW: Missingness-aware split rule that explicitly handles missing values
    as a separate category, similar to bartMachine's approach.
    
    This split rule creates a three-way split:
    1. Values <= split_value go to left child
    2. Values > split_value go to right child  
    3. Missing values (NaN) go to a designated child (left or right)
    
    The missing value routing is determined during split creation and stored as part
    of the split information.
    """

    @staticmethod
    def get_split_value(available_splitting_values):
        """
        [NA-handling] Get split value and missing value routing decision.
        
        Returns:
        - split_value: The threshold for non-missing values
        - missing_goes_left: Boolean indicating whether missing values go to left child
        """
        split_value = None
        missing_goes_left = None
        
        # Filter out NaN values for split value selection
        non_nan_values = available_splitting_values[~np.isnan(available_splitting_values)]
        nan_present = np.any(np.isnan(available_splitting_values))
        
        if non_nan_values.size > 1:
            idx_selected_splitting_values = int(
                np.random.random() * len(non_nan_values)
            )
            split_value = non_nan_values[idx_selected_splitting_values]
            
            # [NA-handling] Decide where missing values should go
            if nan_present:
                missing_goes_left = np.random.choice([True, False])
            else:
                missing_goes_left = True  # Default if no missing values
                
        return split_value, missing_goes_left

    @staticmethod
    def divide(available_splitting_values, split_info):
        """
        [NA-handling] Divide data based on split value and missing value routing.
        
        Args:
        - available_splitting_values: Array of values to split on
        - split_info: Tuple of (split_value, missing_goes_left)
        
        Returns:
        - Boolean array indicating which values go to the left child
        """
        split_value, missing_goes_left = split_info
        
        # Handle case where split_value is None (no valid split found)
        if split_value is None:
            # If no valid split, all values go to the left child
            return np.ones_like(available_splitting_values, dtype=bool)
        
        # Handle non-missing values
        result = available_splitting_values <= split_value
        
        # [NA-handling] Handle missing values according to the routing decision
        nan_mask = np.isnan(available_splitting_values)
        result[nan_mask] = missing_goes_left
        
        return result


class MissingnessAwareCategoricalSplitRule(SplitRule):
    """
    [NA-handling] NEW: Missingness-aware split rule for categorical variables.
    
    This split rule handles categorical variables with missing values by:
    1. Creating a subset of categories that go to the left child
    2. Explicitly deciding whether missing values go to the left or right child
    
    This is particularly useful for categorical variables where missingness
    patterns may be informative.
    """

    @staticmethod
    def get_split_value(available_splitting_values):
        """
        [NA-handling] Get categorical subset and missing value routing decision.
        
        Returns:
        - category_subset: Array of categories that go to the left child
        - missing_goes_left: Boolean indicating whether missing values go to left child
        """
        category_subset = None
        missing_goes_left = None
        
        # Filter out NaN values for category selection
        non_nan_values = available_splitting_values[~np.isnan(available_splitting_values)]
        nan_present = np.any(np.isnan(available_splitting_values))
        
        if non_nan_values.size > 1 and not np.all(
            non_nan_values == non_nan_values[0]
        ):
            unique_values = np.unique(non_nan_values)
            while True:
                sample = np.random.randint(0, 2, size=len(unique_values)).astype(bool)
                if np.any(sample):
                    break
            category_subset = unique_values[sample]
            
            # [NA-handling] Decide where missing values should go
            if nan_present:
                missing_goes_left = np.random.choice([True, False])
            else:
                missing_goes_left = True  # Default if no missing values
                
        return category_subset, missing_goes_left

    @staticmethod
    def divide(available_splitting_values, split_info):
        """
        [NA-handling] Divide categorical data based on subset and missing value routing.
        
        Args:
        - available_splitting_values: Array of categorical values to split on
        - split_info: Tuple of (category_subset, missing_goes_left)
        
        Returns:
        - Boolean array indicating which values go to the left child
        """
        category_subset, missing_goes_left = split_info
        
        # Handle case where category_subset is None (no valid split found)
        if category_subset is None:
            # If no valid split, all values go to the left child
            return np.ones_like(available_splitting_values, dtype=bool)
        
        # Handle non-missing values
        result = np.isin(available_splitting_values, category_subset)
        
        # [NA-handling] Handle missing values according to the routing decision
        nan_mask = np.isnan(available_splitting_values)
        result[nan_mask] = missing_goes_left
        
        return result
