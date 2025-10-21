# Copy this entire script into a Jupyter cell to test the BART fixes

import numpy as np
import pymc as pm
import pymc_bart as pmb

# Create simple simulated data
np.random.seed(42)
n_samples = 100
n_features = 5

# Create features with some missing values
X = np.random.randn(n_samples, n_features)
# Add some missing values (about 10% missing)
X[np.random.choice(n_samples, 10), np.random.choice(n_features, 10)] = np.nan

# Create target variable (without NaN values)
Y = np.exp(0.1 * X[:, 0] + 0.2 * X[:, 1] + 0.3 * X[:, 2] + np.random.randn(n_samples) * 0.1)

# Filter out rows with NaN values in Y (to avoid PyMC Data issues)
valid_mask = ~np.isnan(Y)
X = X[valid_mask]
Y = Y[valid_mask]

print(f"Data shapes after filtering: X={X.shape}, Y={Y.shape}")
print(f"Missing values in X: {np.isnan(X).sum()}")
print(f"Missing values in Y: {np.isnan(Y).sum()}")

print(f"Data shapes after filtering: X={X.shape}, Y={Y.shape}")
print(f"Missing values in X: {np.isnan(X).sum()}")

# Split into train/test
train_idx = np.random.choice(X.shape[0], 80, replace=False)
test_idx = np.setdiff1d(np.arange(X.shape[0]), train_idx)

X_train = X[train_idx]
Y_train = Y[train_idx]
X_test = X[test_idx]
Y_test = Y[test_idx]

print(f"\nTrain: X={X_train.shape}, Y={Y_train.shape}")
print(f"Test: X={X_test.shape}, Y={Y_test.shape}")

# Create split rules for missingness-aware handling
from pymc_bart.split_rules import MissingnessAwareSplitRule, MissingnessAwareCategoricalSplitRule

split_rules = []
for col in range(X_train.shape[1]):
    unique_values = np.unique(X_train[:, col])
    non_nan_values = unique_values[~np.isnan(unique_values)]
    
    if len(non_nan_values) < 10:
        split_rules.append(MissingnessAwareCategoricalSplitRule())
    else:
        split_rules.append(MissingnessAwareSplitRule())

print(f"\nSplit rules created: {len(split_rules)}")

# Test 1: Model Creation
print("\n" + "="*50)
print("TEST 1: Model Creation")
print("="*50)

try:
    with pm.Model() as model:
        X_data = pmb.MutableDataWithNA("X_data", X_train)
        Y_data = pm.Data("Y_data", Y_train)
        
        μ = pmb.BART(
            "μ", 
            X_data.get_value(), 
            pm.math.log(Y_data), 
            m=50,  # Small number for testing
            alpha=0.95, 
            beta=2, 
            split_rules=split_rules, 
            missingness_handling='aware'
        )
        
        s0 = np.median(np.abs(np.log(Y_train) - np.median(np.log(Y_train)))) * 1.4826
        a = 3.5
        b = (a - 1) * (s0**2)
        s_sq = pm.InverseGamma("sigma_sq", alpha=a, beta=b)
        sigma = pm.Deterministic("sigma", pm.math.sqrt(s_sq))
        
        y = pm.Normal(
            "y", 
            mu=μ,
            sigma=sigma, 
            observed=pm.math.log(Y_data)
        )
        
        print("✅ Model created successfully!")
        
except Exception as e:
    print(f"❌ Model creation failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Sampling (just a few samples)
print("\n" + "="*50)
print("TEST 2: Sampling")
print("="*50)

try:
    with model:
        trace = pm.sample(10, tune=5, return_inferencedata=True)
        print("✅ Sampling completed successfully!")
        print(f"Trace shape: {trace.posterior['μ'].shape}")
        
except Exception as e:
    print(f"❌ Sampling failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Prediction
print("\n" + "="*50)
print("TEST 3: Prediction")
print("="*50)

try:
    with model:
        # Update data for prediction using pm.set_data
        pm.set_data({"μ_X": X_test, "Y_data": Y_test})
        
        ppc = pm.sample_posterior_predictive(
            trace, 
            extend_inferencedata=True, 
            predictions=True, 
            random_seed=42
        )
        print("✅ Prediction completed successfully!")
        print(f"Prediction shape: {ppc.predictions['y'].shape}")
        
except Exception as e:
    print(f"❌ Prediction failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("TESTING COMPLETE")
print("="*50)
