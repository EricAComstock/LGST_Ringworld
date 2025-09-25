"""
Benchmark script to compare the original and NumPy-optimized trajectory classification.
This script verifies that both implementations produce identical results and measures performance.
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Import both implementations
from TrajectoryClassification import classify_trajectory as original_classify
from TrajectoryClassification_numpy import classify_trajectory as numpy_classify

def generate_test_cases(n_cases: int = 5) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
    """Generate test cases with different trajectory patterns."""
    test_cases = []
    
    # Common parameters - using the same values as in the implementations
    alpha = 149597652691
    beta = 5000000.0
    y_floor = 149597870691
    
    # Generate trajectories with different patterns
    for i in range(n_cases):
        # Common time array for all coordinates
        t = np.linspace(0, 10, 1000)
        
        # Different patterns for different test cases
        if i == 0:
            # Simple escape: crosses beta boundary and stays out
            x = 1000 * np.sin(t)
            y = alpha + 1000 * np.ones_like(t)  # Stay above alpha
            z = beta * 1.1 * np.ones_like(t)  # Stay outside beta
            expected = {'beta_crossings': 1, 'result': 'escaped'}
            
        elif i == 1:
            # Simple resimulate: stays within beta and above alpha
            x = 1000 * np.sin(t)
            y = alpha + 1000 * np.ones_like(t)  # Stay above alpha
            z = beta * 0.5 * np.sin(t)  # Oscillate within beta
            expected = {'beta_crossings': 0, 'result': 'resimulate'}
            
        elif i == 2:
            # Simple recapture: stays within beta but goes below alpha
            x = 1000 * np.sin(t)
            y = alpha - 1000 * np.ones_like(t)  # Stay below alpha
            z = beta * 0.5 * np.sin(t)  # Stay within beta
            expected = {'beta_crossings': 0, 'result': 'recaptured'}
            
        elif i == 3:
            # Multiple beta crossings, ends outside
            x = 1000 * np.sin(t)
            y = alpha + 1000 * np.ones_like(t)  # Stay above alpha
            z = beta * (1.0 + 0.5 * np.sin(4 * np.pi * t / 10))  # Cross beta multiple times
            expected = {'beta_crossings': 4, 'result': 'escaped'}
            
        else:  # i == 4
            # Complex pattern: crosses beta multiple times, ends inside
            t = np.linspace(0, 10, 1000)
            x = 2000 * np.sin(t)
            y = alpha - 1000 * np.ones_like(t)  # Stay below alpha
            z = beta * (0.8 + 0.4 * np.sin(0.5 * t))  # Oscillate within beta
            expected = {'beta_crossings': 0, 'result': 'recaptured'}
        
        # Ensure all arrays have the same length
        min_len = min(len(x), len(y), len(z))
        trajectory = np.column_stack((x[:min_len], y[:min_len], z[:min_len]))
        test_cases.append((trajectory, expected))
    
    return test_cases

def verify_correctness(test_cases: List[Tuple[np.ndarray, Dict[str, Any]]]) -> bool:
    """
    Verify that both implementations produce the same results.
    
    Returns:
        bool: True if all test cases pass, False otherwise
    """
    print("\n=== Verifying Correctness ===")
    all_correct = True
    
    # Set consistent parameters
    alpha = 149597652691  # y_floor - (218 * 1000)
    beta = 5000000.0
    y_floor = 149597870691
    
    for i, (traj, expected) in enumerate(test_cases):
        # Convert to DataFrame for original implementation
        df = pd.DataFrame(traj, columns=['x', 'y', 'z'])
        
        try:
            # Get results from both implementations
            orig_result = original_classify(alpha, beta, y_floor, df)
            numpy_result = numpy_classify(alpha, beta, y_floor, traj)
            
            # Convert numpy types to Python native types for comparison
            orig_result = (int(orig_result[0]), orig_result[1])
            numpy_result = (int(numpy_result[0]), numpy_result[1])
            
            # Check if results match
            if orig_result != numpy_result:
                print(f"❌ Test case {i+1} - MISMATCH")
                print(f"  Original: beta_crossings={orig_result[0]}, result='{orig_result[1]}'")
                print(f"  NumPy:    beta_crossings={numpy_result[0]}, result='{numpy_result[1]}'")
                all_correct = False
            else:
                print(f"✅ Test case {i+1} - PASSED")
                
            # Compare with expected values if available
            if expected['beta_crossings'] != 'variable':
                if orig_result[0] != expected['beta_crossings']:
                    print(f"  Warning: Expected {expected['beta_crossings']} beta crossings, got {orig_result[0]}")
                if orig_result[1] != expected['result']:
                    print(f"  Warning: Expected result '{expected['result']}', got '{orig_result[1]}'")
                    
        except Exception as e:
            print(f"❌ Test case {i+1} - ERROR: {str(e)}")
            all_correct = False
    
    if all_correct:
        print("\n✅ All test cases produce identical results between implementations!")
    else:
        print("\n❌ Some test cases produced different results between implementations!")
    
    return all_correct

def run_benchmark(test_cases: List[Tuple[np.ndarray, Dict[str, Any]]], num_runs: int = 100) -> Dict[str, Any]:
    """
    Run performance benchmark on both implementations.
    
    Args:
        test_cases: List of test cases to benchmark
        num_runs: Number of runs per test case
        
    Returns:
        Dictionary containing timing results and speedup
    """
    results = {
        'original_times': [],
        'numpy_times': [],
        'speedup': []
    }
    
    # Set consistent parameters
    alpha = 149597652691  # y_floor - (218 * 1000)
    beta = 5000000.0
    y_floor = 149597870691
    
    print("\n=== Running Performance Benchmark ===")
    print(f"Running {num_runs} iterations per test case...")
    
    for i, (traj, _) in enumerate(test_cases):
        # Convert to DataFrame for original implementation
        df = pd.DataFrame(traj, columns=['x', 'y', 'z'])
        
        # Warm-up runs
        for _ in range(5):
            original_classify(alpha, beta, y_floor, df)
            numpy_classify(alpha, beta, y_floor, traj)
        
        # Time original implementation
        start_time = time.perf_counter()
        for _ in range(num_runs):
            original_classify(alpha, beta, y_floor, df)
        orig_time = time.perf_counter() - start_time
        
        # Time NumPy implementation
        start_time = time.perf_counter()
        for _ in range(num_runs):
            numpy_classify(alpha, beta, y_floor, traj)
        numpy_time = time.perf_counter() - start_time
        
        # Calculate speedup
        speedup = orig_time / numpy_time if numpy_time > 0 else float('inf')
        
        results['original_times'].append(orig_time)
        results['numpy_times'].append(numpy_time)
        results['speedup'].append(speedup)
        
        print(f"\nTest case {i+1}:")
        print(f"  Original: {orig_time*1000/num_runs:.4f} ms/iter")
        print(f"  NumPy:    {numpy_time*1000/num_runs:.4f} ms/iter")
        print(f"  Speedup:  {speedup:.2f}x")
    
    return results

def plot_results(test_cases: List[Tuple[np.ndarray, Dict[str, Any]]], results: Dict[str, List[float]]) -> None:
    """
    Plot the benchmark results.
    
    Args:
        test_cases: List of test cases
        results: Dictionary containing timing results
    """
    # Create figure and axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
    
    # Plot execution times
    x = np.arange(len(test_cases))
    width = 0.35
    
    # Convert to milliseconds per iteration
    orig_times = np.array(results['original_times']) * 1000
    numpy_times = np.array(results['numpy_times']) * 1000
    
    # Plot execution times
    bars1 = ax1.bar(x - width/2, orig_times, width, label='Original', color='#1f77b4')
    bars2 = ax1.bar(x + width/2, numpy_times, width, label='NumPy Optimized', color='#ff7f0e')
    
    ax1.set_xlabel('Test Case', fontsize=12)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title('Execution Time per Test Case', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Case {i+1}' for i in range(len(test_cases))])
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels
    def autolabel(bars, ax):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.2f}ms',
                   ha='center', va='bottom', fontsize=8)
    
    autolabel(bars1, ax1)
    autolabel(bars2, ax1)
    
    # Plot speedup
    colors = ['green' if s >= 1 else 'red' for s in results['speedup']]
    bars = ax2.bar(x, results['speedup'], color=colors, alpha=0.7)
    ax2.axhline(y=1, color='r', linestyle='--')
    
    ax2.set_xlabel('Test Case', fontsize=12)
    ax2.set_ylabel('Speedup (x)', fontsize=12)
    ax2.set_title('Speedup of NumPy Implementation', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Case {i+1}' for i in range(len(test_cases))])
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, (bar, speedup) in enumerate(zip(bars, results['speedup'])):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{speedup:.1f}x',
                ha='center', va='bottom', fontsize=9,
                color='green' if speedup >= 1 else 'red')
    
    # Add a table with the results
    cell_text = []
    for i in range(len(test_cases)):
        cell_text.append([
            f"{results['original_times'][i]*1000:.2f} ms",
            f"{results['numpy_times'][i]*1000:.2f} ms",
            f"{results['speedup'][i]:.2f}x"
        ])
    
    table = plt.table(cellText=cell_text,
                     colLabels=['Original', 'NumPy', 'Speedup'],
                     rowLabels=[f'Case {i+1}' for i in range(len(test_cases))],
                     loc='bottom',
                     bbox=[0.1, -0.5, 0.8, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)  # Make room for the table
    
    # Save figure with high DPI
    plt.savefig('trajectory_classification_benchmark.png', dpi=300, bbox_inches='tight')
    print("\n✅ Benchmark results saved to 'trajectory_classification_benchmark.png'")

if __name__ == "__main__":
    print("=== Trajectory Classification Benchmark ===")
    print("This script compares the original and NumPy-optimized implementations.")
    print("It verifies correctness and measures performance improvements.\n")
    
    try:
        # Generate test cases
        print("Generating test cases...")
        test_cases = generate_test_cases(n_cases=5)
        
        # Verify correctness first
        print("\n=== Starting Verification ===")
        all_correct = verify_correctness(test_cases)
        
        if not all_correct:
            print("\n❌ Correctness verification failed. Fix the implementation before benchmarking.")
            exit(1)
        
        # Run benchmark
        results = run_benchmark(test_cases, num_runs=100)
        
        # Plot results
        print("\n=== Generating Plots ===")
        plot_results(test_cases, results)
        
        # Calculate and display overall statistics
        avg_speedup = np.mean(results['speedup'])
        min_speedup = np.min(results['speedup'])
        max_speedup = np.max(results['speedup'])
        
        print("\n=== Benchmark Summary ===")
        print(f"Average speedup: {avg_speedup:.2f}x")
        print(f"Minimum speedup: {min_speedup:.2f}x")
        print(f"Maximum speedup: {max_speedup:.2f}x")
        
        print("\n✅ Benchmark completed successfully!")
        print("\nTo use the optimized version in your code, replace:")
        print("  from TrajectoryClassification import classify_trajectory")
        print("with:")
        print("  from TrajectoryClassification_numpy import classify_trajectory")
        
    except Exception as e:
        print(f"\n❌ An error occurred during benchmarking: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
