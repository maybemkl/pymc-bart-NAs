#!/usr/bin/env python3
"""Test if BART variables are recognized after fixing the dist method."""

import numpy as np
import pymc as pm
import pymc_bart as pmb
from pymc_bart.utils import MutableDataWithNA

# Create simple test data
np.random.seed(42)
X_train = np.random.randn(100, 3)
Y_train = np.random.randn(100)

# Test if BART variables are recognized
with pm.Model() as model:
    X_data = pmb.MutableDataWithNA("X_data", X_train)
    Y_data = pm.Data("Y_data", Y_train)
    
    μ = pmb.BART("μ", X_data.get_value(), pm.math.log(Y_data), 
                  m=10, alpha=0.95, beta=2, 
                  missingness_handling='enhanced')
    
    # Check if PGBART recognizes this variable
    try:
        step = pmb.PGBART([μ])
        print("SUCCESS: PGBART recognized BART variable")
        print(f"μ type: {type(μ)}")
        print(f"μ owner op: {μ.owner.op if μ.owner else 'No owner'}")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

