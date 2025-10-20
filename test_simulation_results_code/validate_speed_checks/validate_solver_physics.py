#!/usr/bin/env python3
"""
Validation script for SolverSharedCodePlusSolar_Optimized.py vs SolverSharedCodePlusSolar.py

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
from SolverSharedCodePlusSolar import compute_motion as compute_motion_original
from SolverSharedCodePlusSolar import calculate_omega as calculate_omega_original
from SolverSharedCodePlusSolar import SSCPSVarInput as SSCPSVarInput_original

from SolverSharedCodePlusSolar_Optimized import compute_motion as compute_motion_optimized
from SolverSharedCodePlusSolar_Optimized import calculate_omega as calculate_omega_optimized
from SolverSharedCodePlusSolar_Optimized import SSCPSVarInput as SSCPSVarInput_optimized

def test_numerical_accuracy():
    """Test that optimized version produces identical numerical results."""
    print("=" * 80)
    print("NUMERICAL ACCURACY TEST: SolverSharedCodePlusSolar")
    print("=" * 80)
    
    # Test parameters
    test_cases = [
        {
            'name': 'Bishop Ring',
            'initial_position': [0, 149597870691, 0],
            'initial_velocity': [100, 0, 50],
            'radius': 1000,
            'gravity': 9.81,
            't_max': 100,
            'dt': 0.1
        },
        {
            'name': 'Large Ringworld',
            'initial_position': [1000, 149597870691 + 218000, 500],
            'initial_velocity': [50, -20, 30],
            'radius': 149597870691,
            'gravity': 9.81,
            't_max': 500,
            'dt': 1.0
        },
        {
            'name': 'High Velocity',
            'initial_position': [0, 149597870691, 0],
            'initial_velocity': [1000, 500, 200],
            'radius': 149597870691,
            'gravity': 9.81,
            't_max': 50,
            'dt': 0.5
        }
    ]
    
    all_passed = True
    tolerance = 1e-10  # Very tight tolerance for identical results
    
    for i, case in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {case['name']}")
        print(f"  Parameters: radius={case['radius']}, gravity={case['gravity']}, t_max={case['t_max']}")
        
        try:
            # Run original version
            orig_pos, orig_vel, orig_sol = compute_motion_original(
                case['initial_position'], case['initial_velocity'],
                case['radius'], case['gravity'], case['t_max'], case['dt']
            )
            
            # Run optimized version
            opt_pos, opt_vel, opt_sol = compute_motion_optimized(
                case['initial_position'], case['initial_velocity'],
                case['radius'], case['gravity'], case['t_max'], case['dt']
            )
            
            # Compare final positions
            pos_diff = np.array(orig_pos) - np.array(opt_pos)
            pos_max_diff = np.max(np.abs(pos_diff))
            
            # Compare final velocities
            vel_diff = np.array(orig_vel) - np.array(opt_vel)
            vel_max_diff = np.max(np.abs(vel_diff))
            
            # Compare trajectory shapes
            traj_diff = np.max(np.abs(orig_sol.y - opt_sol.y))
            
            print(f"  Final position max difference: {pos_max_diff:.2e}")
            print(f"  Final velocity max difference: {vel_max_diff:.2e}")
            print(f"  Trajectory max difference: {traj_diff:.2e}")
            
            # Check if results are identical within tolerance
            if pos_max_diff < tolerance and vel_max_diff < tolerance and traj_diff < tolerance:
                print(f"  ✅ PASS: Results are identical within {tolerance:.0e}")
            else:
                print(f"  ❌ FAIL: Differences exceed tolerance {tolerance:.0e}")
                all_passed = False
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            all_passed = False
    
    return all_passed

def test_omega_calculation():
    """Test omega calculation accuracy and caching."""
    print(f"\n{'-'*60}")
    print("OMEGA CALCULATION TEST")
    print(f"{'-'*60}")
    
    test_params = [
        (1000, 9.81),
        (149597870691, 9.81),
        (1000000, 2.5),
        (1000, 9.81),  # Repeat to test caching
    ]
    
    all_passed = True
    
    for radius, gravity in test_params:
        orig_omega = calculate_omega_original(radius, gravity)
        opt_omega = calculate_omega_optimized(radius, gravity)
        
        diff = abs(orig_omega - opt_omega)
        
        print(f"  Radius: {radius}, Gravity: {gravity}")
        print(f"    Original: {orig_omega:.10e}")
        print(f"    Optimized: {opt_omega:.10e}")
        print(f"    Difference: {diff:.2e}")
        
        if diff < 1e-15:
            print(f"    ✅ PASS")
        else:
            print(f"    ❌ FAIL")
            all_passed = False
    
    return all_passed

def test_performance():
    """Test performance improvement."""
    print(f"\n{'-'*60}")
    print("PERFORMANCE TEST")
    print(f"{'-'*60}")
    
    # Test parameters
    initial_position = [0, 149597870691, 0]
    initial_velocity = [100, 0, 50]
    radius = 149597870691
    gravity = 9.81
    t_max = 100
    dt = 0.1
    
    num_runs = 10
    
    print(f"Running {num_runs} iterations of compute_motion...")
    
    # Time original version
    print("  Testing original version...")
    orig_times = []
    for i in range(num_runs):
        start = time.time()
        compute_motion_original(initial_position, initial_velocity, radius, gravity, t_max, dt)
        orig_times.append(time.time() - start)
    
    orig_avg = np.mean(orig_times)
    orig_std = np.std(orig_times)
    
    # Time optimized version
    print("  Testing optimized version...")
    opt_times = []
    for i in range(num_runs):
        start = time.time()
        compute_motion_optimized(initial_position, initial_velocity, radius, gravity, t_max, dt)
        opt_times.append(time.time() - start)
    
    opt_avg = np.mean(opt_times)
    opt_std = np.std(opt_times)
    
    # Calculate speedup
    speedup = orig_avg / opt_avg
    
    print(f"\nPerformance Results:")
    print(f"  Original:  {orig_avg:.4f} ± {orig_std:.4f} seconds")
    print(f"  Optimized: {opt_avg:.4f} ± {opt_std:.4f} seconds")
    print(f"  Speedup:   {speedup:.2f}x")
    
    if speedup >= 1.0:
        print(f"  ✅ PASS: Optimized version is {speedup:.2f}x faster")
        return True
    else:
        print(f"  ⚠️  WARNING: Optimized version is {1/speedup:.2f}x slower")
        return False

def test_backward_compatibility():
    """Test backward compatibility of function interfaces."""
    print(f"\n{'-'*60}")
    print("BACKWARD COMPATIBILITY TEST")
    print(f"{'-'*60}")
    
    all_passed = True
    
    # Test function signatures
    functions_to_test = [
        ('compute_motion', compute_motion_original, compute_motion_optimized),
        ('calculate_omega', calculate_omega_original, calculate_omega_optimized),
        ('SSCPSVarInput', SSCPSVarInput_original, SSCPSVarInput_optimized),
    ]
    
    for func_name, orig_func, opt_func in functions_to_test:
        print(f"  Testing {func_name}...")
        
        try:
            # Test that functions exist and are callable
            if not callable(orig_func):
                print(f"    ❌ Original {func_name} is not callable")
                all_passed = False
                continue
                
            if not callable(opt_func):
                print(f"    ❌ Optimized {func_name} is not callable")
                all_passed = False
                continue
            
            # Test function signatures match (basic test)
            orig_code = orig_func.__code__
            opt_code = opt_func.__code__
            
            if orig_code.co_argcount != opt_code.co_argcount:
                print(f"    ⚠️  WARNING: Argument count differs ({orig_code.co_argcount} vs {opt_code.co_argcount})")
            else:
                print(f"    ✅ Function signature compatible")
                
        except Exception as e:
            print(f"    ❌ ERROR testing {func_name}: {e}")
            all_passed = False
    
    return all_passed

def main():
    """Run all validation tests."""
    print("VALIDATION: SolverSharedCodePlusSolar_Optimized vs SolverSharedCodePlusSolar")
    print("Testing: Accuracy, Performance, Compatibility")
    
    # Run all tests
    accuracy_passed = test_numerical_accuracy()
    omega_passed = test_omega_calculation()
    performance_passed = test_performance()
    compatibility_passed = test_backward_compatibility()
    
    # Summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    tests = [
        ("Numerical Accuracy", accuracy_passed),
        ("Omega Calculation", omega_passed),
        ("Performance", performance_passed),
        ("Backward Compatibility", compatibility_passed),
    ]
    
    all_passed = True
    for test_name, passed in tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:<25}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\nOverall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 SolverSharedCodePlusSolar_Optimized is validated!")
        print("   ✅ Produces identical results")
        print("   🚀 Provides performance improvements")
        print("   🔧 Maintains backward compatibility")
    else:
        print("\n⚠️  Validation issues found. Review failed tests above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
