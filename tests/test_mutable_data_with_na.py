"""
Tests for MutableDataWithNA class and its integration with PyMC's Data system.
"""

import numpy as np
import pytest
import pymc as pm
import pymc_bart as pmb
from pymc_bart.utils import MutableDataWithNA, create_mutable_data_with_na


class TestMutableDataWithNA:
    """Test cases for MutableDataWithNA class."""
    
    def test_basic_creation(self):
        """Test basic creation of MutableDataWithNA."""
        X = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])
        data_X = MutableDataWithNA("X", X)
        
        assert data_X.name == "X"
        assert data_X.shape == (3, 3)
        assert data_X.dtype == np.float64
        # Use np.allclose for NaN-aware comparison
        assert np.allclose(data_X.get_value(), X, equal_nan=True)
    
    def test_creation_with_different_dtypes(self):
        """Test creation with different data types."""
        # Integer input should be converted to float
        X_int = np.array([[1, 2, 3], [4, 5, 6]])
        data_X = MutableDataWithNA("X", X_int)
        assert data_X.dtype == np.float64
        
        # Float32 input should be preserved
        X_float32 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        data_X = MutableDataWithNA("X", X_float32)
        assert data_X.dtype == np.float32
    
    def test_set_value(self):
        """Test updating values."""
        X = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0]])
        data_X = MutableDataWithNA("X", X)
        
        new_X = np.array([[10.0, 20.0, np.nan], [40.0, 50.0, 60.0]])
        data_X.set_value(new_X)
        
        assert np.allclose(data_X.get_value(), new_X, equal_nan=True)
    
    def test_set_value_shape_mismatch(self):
        """Test that setting value with wrong shape raises error."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        data_X = MutableDataWithNA("X", X)
        
        new_X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with pytest.raises(ValueError, match="shape"):
            data_X.set_value(new_X)
    
    def test_array_interface(self):
        """Test numpy array interface."""
        X = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0]])
        data_X = MutableDataWithNA("X", X)
        
        # Test __array__ method
        arr = np.array(data_X)
        assert np.allclose(arr, X, equal_nan=True)
        
        # Test indexing
        assert data_X[0, 0] == 1.0
        assert np.isnan(data_X[0, 2])
        
        # Test len
        assert len(data_X) == 2
    
    def test_convenience_function(self):
        """Test the convenience function create_mutable_data_with_na."""
        X = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0]])
        data_X = create_mutable_data_with_na("X", X)
        
        assert isinstance(data_X, MutableDataWithNA)
        assert data_X.name == "X"
        assert np.allclose(data_X.get_value(), X, equal_nan=True)


class TestMutableDataWithNAAndPyMC:
    """Test integration with PyMC's Data system."""
    
    def test_with_pm_data(self):
        """Test using MutableDataWithNA directly with BART (bypassing pm.Data)."""
        X = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])
        Y = np.array([1.0, 2.0, 3.0])
        
        # Create mutable data container
        data_X = MutableDataWithNA("X", X)
        
        with pm.Model() as model:
            # Use MutableDataWithNA directly with BART (bypassing pm.Data for missing values)
            mu = pmb.BART("mu", data_X.get_value(), Y, m=2, missingness_handling="aware")
            sigma = pm.HalfNormal("sigma", 1)
            y = pm.Normal("y", mu, sigma, observed=Y, shape=mu.shape)
            
            # Model should compile without errors
            assert model is not None
    
    def test_data_update_during_sampling(self):
        """Test updating data during sampling."""
        X = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])
        Y = np.array([1.0, 2.0, 3.0])
        
        data_X = MutableDataWithNA("X", X)
        
        with pm.Model() as model:
            # Use MutableDataWithNA directly with BART
            mu = pmb.BART("mu", data_X.get_value(), Y, m=2, missingness_handling="aware")
            sigma = pm.HalfNormal("sigma", 1)
            y = pm.Normal("y", mu, sigma, observed=Y, shape=mu.shape)
            
            # Sample
            idata = pm.sample(tune=50, draws=50, chains=1, random_seed=3415)
            
            # Update data for posterior predictive (same shape as original)
            new_X = np.array([[10.0, 20.0, np.nan], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]])
            data_X.set_value(new_X)
            
            # Should be able to sample posterior predictive with new data
            ppc = pm.sample_posterior_predictive(idata)
            assert ppc is not None
    
    def test_missingness_aware_bart_with_mutable_data(self):
        """Test BART with missingness-aware handling using MutableDataWithNA."""
        # Create data with missing values
        X = np.array([
            [1.0, 2.0, np.nan, 4.0],
            [5.0, np.nan, 7.0, 8.0],
            [9.0, 10.0, 11.0, np.nan],
            [13.0, 14.0, 15.0, 16.0]
        ])
        Y = np.array([1.0, 2.0, 3.0, 4.0])
        
        data_X = MutableDataWithNA("X", X)
        
        with pm.Model() as model:
            # Test different missingness handling strategies
            for i, strategy in enumerate(["enhanced", "aware"]):
                mu = pmb.BART(
                    f"mu_{strategy}", 
                    data_X.get_value(), 
                    Y, 
                    m=2, 
                    missingness_handling=strategy,
                    auto_detect_categorical=True
                )
                sigma = pm.HalfNormal(f"sigma_{strategy}", 1)
                y = pm.Normal(f"y_{strategy}", mu, sigma, observed=Y, shape=mu.shape)
                
                # Model should compile without errors
                assert model is not None


class TestMutableDataWithNAEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_array(self):
        """Test with empty array."""
        X = np.array([])
        data_X = MutableDataWithNA("X", X)
        assert data_X.shape == (0,)
    
    def test_single_value(self):
        """Test with single value."""
        X = np.array([1.0])
        data_X = MutableDataWithNA("X", X)
        assert data_X.shape == (1,)
        assert data_X[0] == 1.0
    
    def test_all_missing(self):
        """Test with array containing only missing values."""
        X = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        data_X = MutableDataWithNA("X", X)
        assert data_X.shape == (2, 2)
        assert np.all(np.isnan(data_X.get_value()))
    
    def test_no_missing(self):
        """Test with array containing no missing values."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        data_X = MutableDataWithNA("X", X)
        assert data_X.shape == (2, 2)
        assert not np.any(np.isnan(data_X.get_value()))


if __name__ == "__main__":
    pytest.main([__file__]) 