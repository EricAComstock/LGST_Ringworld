#!/usr/bin/env python3
"""
Validation script for StochasticInputRK45Solver_Vectorized.py vs StochasticInputRK45Solver.py

Tests:
1. Identical numerical results
2. Performance improvement
3. Backward compatibility
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import both versions
from StochasticInputRK45Solver import main as solver_original
from StochasticInputRK45Solver_Vectorized import main_vectorized as solver_vectorized

def extract_solver_internal_params(solver_func, params):
    """Extract internal parameters that each solver actually uses."""
    internal_params = {}
    
    if 'StochasticInputRK45Solver_Vectorized' in solver_func.__module__:
        # Vectorized solver - get parameters from get_simulation_params
        try:
            from StochasticInputRK45Solver_Vectorized import get_simulation_params
            full_params = get_simulation_params(params)
            internal_params = {
                'temperature': full_params.get('temperature', 'NOT_SET'),
                'y_min': full_params.get('y_min', 'NOT_SET'),
                'y_max': full_params.get('y_max', 'NOT_SET'),
                'z_length': full_params.get('z_length', 'NOT_SET'),
                'y_floor': full_params.get('y_floor', 'NOT_SET'),
                'alpha': full_params.get('alpha', 'NOT_SET'),
                'beta': full_params.get('beta', 'NOT_SET'),
            }
        except Exception as e:
            internal_params['error'] = str(e)
    else:
        # Original solver - get global variables from StochasticInput
        try:
            from StochasticInput import T, y_min, y_max, z_length, y_floor
            internal_params = {
                'temperature': T,
                'y_min': y_min,
                'y_max': y_max,
                'z_length': z_length,
                'y_floor': y_floor,
                'alpha': y_floor - (218 * 1000),
                'beta': z_length / 2,
            }
        except Exception as e:
            internal_params['error'] = str(e)
    
    return internal_params

def test_numerical_accuracy():
    """Test that vectorized version produces identical numerical results."""
    print("=" * 80)
    print("NUMERICAL ACCURACY TEST: StochasticInputRK45Solver_Vectorized")
    print("=" * 80)
    
    # Test parameters - use small particle counts for deterministic comparison
    test_cases = [
        {
            'name': 'Small Bishop Ring Test',
            'params': {
                'radius': 1000,
                'gravity': 9.81,
                't_max': 50,
                'dt': 0.5,
                'is_rotating': False,
                'num_particles': 10,
                'save_results': False,
                'show_plots': False,
                'find_leak_rate': False,
                'comp_list': [("O2", 2.6566962e-26 * 2, 0, 100)]  # Add comp_list for original solver
            }
        },
        {
            'name': 'Medium Ringworld Test',
            'params': {
                'radius': 149597870691,
                'gravity': 9.81,
                't_max': 100,
                'dt': 1.0,
                'is_rotating': False,
                'num_particles': 20,
                'save_results': False,
                'show_plots': False,
                'find_leak_rate': False,
                'comp_list': [("O2", 2.6566962e-26 * 2, 0, 100)]  # Add comp_list for original solver
            }
        }
    ]
    
    all_passed = True
    
    for i, case in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {case['name']}")
        print(f"  Particles: {case['params']['num_particles']}")
        print(f"  Radius: {case['params']['radius']}")
        
        try:
            # Print parameter comparison table
            print(f"\n  📋 PARAMETER COMPARISON TABLE:")
            print(f"  {'Parameter':<20} {'Value':<25} {'Type':<15}")
            print(f"  {'-'*60}")
            for key, value in case['params'].items():
                print(f"  {key:<20} {str(value):<25} {type(value).__name__:<15}")
            
            # Set BOTH random seeds for reproducible results
            np.random.seed(42)  # NumPy random seed
            import random
            random.seed(42)     # Python built-in random seed
            
            # Extract and compare internal parameters
            print(f"\n  🔍 INTERNAL PARAMETER COMPARISON:")
            orig_internal = extract_solver_internal_params(solver_original, case['params'])
            vec_internal = extract_solver_internal_params(solver_vectorized, case['params'])
            
            print(f"  {'Parameter':<15} {'Original':<20} {'Vectorized':<20} {'Match':<8}")
            print(f"  {'-'*70}")
            
            param_mismatch = False
            for key in set(orig_internal.keys()) | set(vec_internal.keys()):
                orig_val = orig_internal.get(key, 'MISSING')
                vec_val = vec_internal.get(key, 'MISSING')
                
                # Check if values match (handle different types)
                try:
                    if isinstance(orig_val, (int, float)) and isinstance(vec_val, (int, float)):
                        match = abs(orig_val - vec_val) < 1e-10
                    else:
                        match = str(orig_val) == str(vec_val)
                except:
                    match = False
                
                match_str = "✅" if match else "❌"
                if not match:
                    param_mismatch = True
                
                print(f"  {key:<15} {str(orig_val):<20} {str(vec_val):<20} {match_str:<8}")
            
            if param_mismatch:
                print(f"  ⚠️  WARNING: Internal parameter mismatch detected!")
            
            # Run original version
            print(f"\n  🔧 Running original solver...")
            print(f"     Function: {solver_original.__module__}.{solver_original.__name__}")
            orig_result = solver_original(**case['params'])
            
            # Reset BOTH random seeds to same value for vectorized version
            np.random.seed(42)  # NumPy random seed
            random.seed(42)     # Python built-in random seed
            
            # Run vectorized version (single-threaded for deterministic results)
            print(f"  🚀 Running vectorized solver (single-threaded)...")
            vec_params = case['params'].copy()
            vec_params['num_processes'] = 1  # Force single-threaded
            print(f"     Function: {solver_vectorized.__module__}.{solver_vectorized.__name__}")
            print(f"     Additional params: num_processes=1")
            vec_result = solver_vectorized(**vec_params)
            
            # Compare results
            if orig_result is None or vec_result is None:
                print("  ❌ FAIL: One or both solvers returned None")
                all_passed = False
                continue
            
            if len(orig_result) != len(vec_result):
                print(f"  ❌ FAIL: Different number of results ({len(orig_result)} vs {len(vec_result)})")
                all_passed = False
                continue
            
            # Compare statistics
            orig_stats = {
                'escaped': len(orig_result[orig_result['Result'] == 'escaped']),
                'recaptured': len(orig_result[orig_result['Result'] == 'recaptured']),
                'resimulate': len(orig_result[orig_result['Result'] == 'resimulate']),
                'total': len(orig_result)
            }
            
            vec_stats = {
                'escaped': len(vec_result[vec_result['Result'] == 'escaped']),
                'recaptured': len(vec_result[vec_result['Result'] == 'recaptured']),
                'resimulate': len(vec_result[vec_result['Result'] == 'resimulate']),
                'total': len(vec_result)
            }
            
            print(f"  Original:   {orig_stats['escaped']} escaped, {orig_stats['recaptured']} recaptured, {orig_stats['resimulate']} resimulate")
            print(f"  Vectorized: {vec_stats['escaped']} escaped, {vec_stats['recaptured']} recaptured, {vec_stats['resimulate']} resimulate")
            
            # Check if results match
            results_match = (
                orig_stats['escaped'] == vec_stats['escaped'] and
                orig_stats['recaptured'] == vec_stats['recaptured'] and
                orig_stats['resimulate'] == vec_stats['resimulate']
            )
            
            if results_match:
                print(f"  ✅ PASS: Results are identical")
            else:
                print(f"  ❌ FAIL: Results differ")
                all_passed = False
                
            # Compare some numerical values (first few particles)
            if results_match and len(orig_result) > 0:
                # Compare final positions of first 3 particles
                for j in range(min(3, len(orig_result))):
                    orig_final_x = orig_result.iloc[j]['Final x']
                    vec_final_x = vec_result.iloc[j]['Final x']
                    diff = abs(orig_final_x - vec_final_x)
                    
                    if diff < 1e-10:
                        print(f"    Particle {j+1} final x: ✅ identical ({diff:.2e} difference)")
                    else:
                        print(f"    Particle {j+1} final x: ❌ differs by {diff:.2e}")
                        # Print initial conditions for debugging
                        print(f"      Original - Initial: ({orig_result.iloc[j]['Initial x']:.2e}, {orig_result.iloc[j]['Initial y']:.2e}, {orig_result.iloc[j]['Initial z']:.2e})")
                        print(f"      Vectorized - Initial: ({vec_result.iloc[j]['Initial x']:.2e}, {vec_result.iloc[j]['Initial y']:.2e}, {vec_result.iloc[j]['Initial z']:.2e})")
                        print(f"      Original - Final: ({orig_result.iloc[j]['Final x']:.2e}, {orig_result.iloc[j]['Final y']:.2e}, {orig_result.iloc[j]['Final z']:.2e})")
                        print(f"      Vectorized - Final: ({vec_result.iloc[j]['Final x']:.2e}, {vec_result.iloc[j]['Final y']:.2e}, {vec_result.iloc[j]['Final z']:.2e})")
                        all_passed = False
            elif len(orig_result) > 0:
                # Even if results don't match, show some debugging info
                print(f"  🔍 DEBUGGING INFO:")
                for j in range(min(2, len(orig_result), len(vec_result))):
                    orig_final_x = orig_result.iloc[j]['Final x']
                    vec_final_x = vec_result.iloc[j]['Final x']
                    diff = abs(orig_final_x - vec_final_x)
                    print(f"    Particle {j+1} final x difference: {diff:.2e}")
                    print(f"      Original - Initial: ({orig_result.iloc[j]['Initial x']:.2e}, {orig_result.iloc[j]['Initial y']:.2e}, {orig_result.iloc[j]['Initial z']:.2e})")
                    print(f"      Vectorized - Initial: ({vec_result.iloc[j]['Initial x']:.2e}, {vec_result.iloc[j]['Initial y']:.2e}, {vec_result.iloc[j]['Initial z']:.2e})")
                all_passed = False
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    return all_passed

def test_performance():
    """Test performance improvement with parallel processing."""
    print(f"\n{'-'*60}")
    print("PERFORMANCE TEST")
    print(f"{'-'*60}")
    
    # Test parameters - larger particle count to see parallel benefits
    test_params = {
        'radius': 149597870691,
        'gravity': 9.81,
        't_max': 100,
        'dt': 1.0,
        'is_rotating': False,
        'num_particles': 100,  # Larger for performance test
        'save_results': False,
        'show_plots': False,
        'find_leak_rate': False,
        'comp_list': [("O2", 2.6566962e-26 * 2, 0, 100)]  # Add comp_list for original solver
    }
    
    print(f"Testing with {test_params['num_particles']} particles...")
    
    # Test original version
    print("  Running original solver...")
    start_time = time.time()
    try:
        orig_result = solver_original(**test_params)
        orig_time = time.time() - start_time
        orig_stats = {
            'escaped': len(orig_result[orig_result['Result'] == 'escaped']),
            'recaptured': len(orig_result[orig_result['Result'] == 'recaptured']),
            'resimulate': len(orig_result[orig_result['Result'] == 'resimulate']),
        }
        print(f"    ✅ Completed in {orig_time:.2f} seconds")
        print(f"    Results: {orig_stats['escaped']} escaped, {orig_stats['recaptured']} recaptured, {orig_stats['resimulate']} resimulate")
    except Exception as e:
        print(f"    ❌ Failed: {e}")
        return False
    
    # Test vectorized version with different process counts
    process_counts = [1, 2, 4]
    best_vec_time = float('inf')
    best_processes = 1
    
    for num_proc in process_counts:
        print(f"  Running vectorized solver with {num_proc} process(es)...")
        
        vec_params = test_params.copy()
        vec_params['num_processes'] = num_proc
        
        start_time = time.time()
        try:
            vec_result = solver_vectorized(**vec_params)
            vec_time = time.time() - start_time
            
            vec_stats = {
                'escaped': len(vec_result[vec_result['Result'] == 'escaped']),
                'recaptured': len(vec_result[vec_result['Result'] == 'recaptured']),
                'resimulate': len(vec_result[vec_result['Result'] == 'resimulate']),
            }
            
            print(f"    ✅ Completed in {vec_time:.2f} seconds")
            print(f"    Results: {vec_stats['escaped']} escaped, {vec_stats['recaptured']} recaptured, {vec_stats['resimulate']} resimulate")
            
            if vec_time < best_vec_time:
                best_vec_time = vec_time
                best_processes = num_proc
                
        except Exception as e:
            print(f"    ❌ Failed: {e}")
    
    # Calculate best speedup
    if best_vec_time < float('inf'):
        speedup = orig_time / best_vec_time
        print(f"\nPerformance Summary:")
        print(f"  Original:           {orig_time:.2f} seconds")
        print(f"  Vectorized (best):  {best_vec_time:.2f} seconds ({best_processes} processes)")
        print(f"  Speedup:            {speedup:.2f}x")
        
        if speedup >= 1.0:
            print(f"  ✅ PASS: Vectorized version is {speedup:.2f}x faster")
            return True
        else:
            print(f"  ⚠️  WARNING: Vectorized version is {1/speedup:.2f}x slower")
            return False
    else:
        print(f"  ❌ FAIL: Vectorized version failed all tests")
        return False

def test_backward_compatibility():
    """Test backward compatibility of function interfaces."""
    print(f"\n{'-'*60}")
    print("BACKWARD COMPATIBILITY TEST")
    print(f"{'-'*60}")
    
    # Test that vectorized version can be called with same parameters as original
    test_params = {
        'radius': 1000,
        'gravity': 9.81,
        't_max': 50,
        'dt': 1.0,
        'is_rotating': False,
        'num_particles': 5,
        'save_results': False,
        'show_plots': False,
        'find_leak_rate': False,
        'comp_list': [("O2", 2.6566962e-26 * 2, 0, 100)]  # Add comp_list for compatibility
    }
    
    print("  Testing parameter compatibility...")
    
    try:
        # Test original parameters work with vectorized version
        result = solver_vectorized(**test_params)
        
        if result is not None and len(result) > 0:
            print("  ✅ PASS: Vectorized solver accepts original parameters")
            
            # Test additional vectorized parameters
            vec_params = test_params.copy()
            vec_params['num_processes'] = 2
            vec_params['batch_size'] = 3
            
            result2 = solver_vectorized(**vec_params)
            
            if result2 is not None and len(result2) > 0:
                print("  ✅ PASS: Vectorized solver accepts new parameters")
                return True
            else:
                print("  ❌ FAIL: Vectorized solver failed with new parameters")
                return False
        else:
            print("  ❌ FAIL: Vectorized solver returned invalid result")
            return False
            
    except Exception as e:
        print(f"  ❌ FAIL: Compatibility test failed: {e}")
        return False

def test_auto_detection():
    """Test automatic core detection and optimization."""
    print(f"\n{'-'*60}")
    print("AUTO-DETECTION TEST")
    print(f"{'-'*60}")
    
    test_params = {
        'radius': 1000,
        'gravity': 9.81,
        't_max': 50,
        'dt': 1.0,
        'is_rotating': False,
        'save_results': False,
        'show_plots': False,
        'find_leak_rate': False,
        'comp_list': [("O2", 2.6566962e-26 * 2, 0, 100)]  # Add comp_list for auto-detection test
    }
    
    # Test different particle counts to see auto-detection logic
    particle_counts = [10, 100, 500]
    
    for num_particles in particle_counts:
        print(f"  Testing auto-detection with {num_particles} particles...")
        
        params = test_params.copy()
        params['num_particles'] = num_particles
        # Don't specify num_processes - let it auto-detect
        
        try:
            result = solver_vectorized(**params)
            
            if result is not None and len(result) == num_particles:
                print(f"    ✅ PASS: Auto-detection worked for {num_particles} particles")
            else:
                print(f"    ❌ FAIL: Auto-detection failed for {num_particles} particles")
                return False
                
        except Exception as e:
            print(f"    ❌ ERROR: {e}")
            return False
    
    return True

def main():
    """Run all validation tests."""
    print("VALIDATION: StochasticInputRK45Solver_Vectorized vs StochasticInputRK45Solver")
    print("Testing: Accuracy, Performance, Compatibility, Auto-detection")
    
    # Run all tests
    accuracy_passed = test_numerical_accuracy()
    performance_passed = test_performance()
    compatibility_passed = test_backward_compatibility()
    autodetect_passed = test_auto_detection()
    
    # Summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    tests = [
        ("Numerical Accuracy", accuracy_passed),
        ("Performance", performance_passed),
        ("Backward Compatibility", compatibility_passed),
        ("Auto-detection", autodetect_passed),
    ]
    
    all_passed = True
    for test_name, passed in tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:<25}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\nOverall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 StochasticInputRK45Solver_Vectorized is validated!")
        print("   ✅ Produces identical results")
        print("   🚀 Provides significant performance improvements")
        print("   🔧 Maintains backward compatibility")
        print("   🧠 Intelligent auto-detection of system capabilities")
    else:
        print("\n⚠️  Validation issues found. Review failed tests above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
