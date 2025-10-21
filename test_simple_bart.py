#!/usr/bin/env python3
"""Simple test to check BART import and basic functionality."""

import sys
print(f"Python path: {sys.path}")

try:
    import pymc_bart as pmb
    print("SUCCESS: pymc_bart imported")
    print(f"Available attributes: {dir(pmb)}")
    
    # Check if MutableDataWithNA is available
    if hasattr(pmb, 'MutableDataWithNA'):
        print("SUCCESS: MutableDataWithNA is available")
    else:
        print("FAILED: MutableDataWithNA not found")
        
    # Check if BART is available
    if hasattr(pmb, 'BART'):
        print("SUCCESS: BART is available")
    else:
        print("FAILED: BART not found")
        
    # Check if PGBART is available
    if hasattr(pmb, 'PGBART'):
        print("SUCCESS: PGBART is available")
    else:
        print("FAILED: PGBART not found")
        
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

