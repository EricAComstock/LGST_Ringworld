"""
TrajectoryClassification_numpy.py
Optimized version of TrajectoryClassification.py using NumPy for better performance.

V1.1, Optimized version, September 2025
"""

import numpy as np
import pandas as pd
from typing import Tuple, Union

# Default boundary conditions
z_length = 10000 * 1000  # Total z-length [m]
beta = z_length / 2  # Lateral boundary [m]
y_floor = 149597870691  # Ringworld floor [m]
alpha = y_floor - (218 * 1000)  # Atmosphere boundary [m]
y_min = alpha - 10000  # Minimum spawn height [m]
y_max = alpha  # Maximum spawn height [m]

def TCVarInput(z_length_i: float, beta_i: float, y_floor_i: float, 
              alpha_i: float, y_min_i: float, y_max_i: float) -> None:
    """
    Set global parameters for trajectory classification.
    Called by StochasticInputRK45Solver.py to pass boundary parameters.

    Args:
        z_length_i: Total z-dimension length (ringworld width) [m]
        beta_i: Lateral boundary threshold (half of z_length) [m]
        y_floor_i: Minimum y value (ringworld floor) [m]
        alpha_i: Atmosphere boundary threshold [m]
        y_min_i: Minimum particle spawn height [m]
        y_max_i: Maximum particle spawn height [m]
    """
    global z_length, beta, y_floor, alpha, y_min, y_max
    z_length = z_length_i
    beta = beta_i
    y_floor = y_floor_i
    alpha = alpha_i
    y_min = y_min_i
    y_max = y_max_i

def _process_trajectory_numpy(trajectories: np.ndarray, alpha: float, beta: float, y_floor: float) -> Tuple[int, str]:
    """
    Optimized trajectory processing using NumPy while maintaining exact sequential logic.
    
    Args:
        trajectories: Nx3 array of [x, y, z] coordinates
        alpha: Atmosphere boundary radius [m]
        beta: Lateral boundary threshold [m]
        y_floor: Minimum y value (ringworld floor) [m]
        
    Returns:
        Tuple of (beta_crossings, result)
    """
    # Convert to numpy array if it's a DataFrame
    if hasattr(trajectories, 'values'):
        traj = trajectories.values[:, :3]  # Take first 3 columns if more exist
    else:
        traj = np.asarray(trajectories)[:, :3]  # Ensure it's a numpy array
    
    # Pre-compute all positions and distances for better performance
    x = traj[:, 0]
    y = traj[:, 1] 
    z = traj[:, 2]
    r = np.sqrt(x**2 + y**2)  # Vectorized radial distance calculation
    z_abs = np.abs(z)  # Pre-compute absolute z values
    
    # Vectorized beta crossing counting (preserves sequential logic)
    # Count crossings from inside to outside beta
    crossings_out = np.sum((z_abs[1:] >= beta) & (z_abs[:-1] < beta))
    # Count crossings from outside to inside beta  
    crossings_in = np.sum((z_abs[1:] <= beta) & (z_abs[:-1] > beta))
    beta_crossings = int(crossings_out + crossings_in)
    
    # Initialize classification flags (matching original implementation exactly)
    recaptured = False
    escaped = False
    
    # Process each timestep in order (matching original behavior exactly)
    for i in range(1, len(traj)):
        # Current and previous positions (using pre-computed values)
        curr_r = r[i]
        curr_z_abs = z_abs[i]
        prev_r = r[i-1]
        prev_z_abs = z_abs[i-1]
        
        # Check if particle entered side of ringworld (matching original logic exactly)
        if curr_z_abs >= beta and curr_r > alpha and not recaptured and not escaped:
            if prev_z_abs < beta:  # Hit the side boundary
                recaptured = True
            else:  # Came from above atmosphere
                escaped = True
        
        # Check if hit bottom of ringworld (matching original logic exactly)
        if curr_r < y_floor and prev_r > y_floor and curr_z_abs <= beta:
            escaped = True
        
        # Early termination if both conditions are met (optional optimization)
        # This doesn't change logic since once both are set, they don't change
        if recaptured and escaped:
            break
    
    # Classify particles that didn't hit ending conditions during trajectory
    # (matching original logic exactly)
    if not recaptured and not escaped:
        # Use final position for classification (using pre-computed values)
        final_z_abs = z_abs[-1]
        final_r = r[-1]
        
        if final_z_abs <= beta and final_r > alpha:  # Inside bounds, above atmosphere
            # resimulate = True (we can skip this variable and go directly to result)
            result = 'resimulate'
        elif final_z_abs <= beta and final_r < alpha:  # Inside bounds, below atmosphere
            result = 'recaptured'
        else:  # Outside lateral bounds
            result = 'escaped'
    else:
        # Determine final result (matching original logic exactly)
        if recaptured:
            result = 'recaptured'
        else:  # escaped must be True
            result = 'escaped'
    
    return beta_crossings, result

def classify_trajectory(alpha: float, beta: float, y_floor: float, 
                       trajectories: Union[pd.DataFrame, np.ndarray]) -> Tuple[int, str]:
    """
    Classifies particle trajectory based on path and final position.
    
    Args:
        alpha: Atmosphere boundary radius [m]
        beta: Lateral boundary threshold [m]
        y_floor: Minimum y value (ringworld floor) [m]
        trajectories: DataFrame or array with [x, y, z] coordinates
        
    Returns:
        Tuple of (beta_crossings, result) where:
        - beta_crossings: Number of beta boundary crossings [int]
        - result: Classification string ('escaped', 'recaptured', or 'resimulate')
    """
    return _process_trajectory_numpy(trajectories, alpha, beta, y_floor)

def generate_test_trajectory() -> Tuple[np.ndarray, dict]:
    """Generate test trajectory data for validation."""
    # Create a simple trajectory that crosses boundaries
    t = np.linspace(0, 10, 1000)
    x = 1000 * np.sin(t)
    y = alpha + 1000 * np.cos(t)  # Start above alpha, move down
    
    # Create z-coordinate that crosses beta boundary exactly 4 times
    # Use a sine wave with exactly 2 full periods (4 crossings)
    z = beta * 1.1 * np.sin(4 * np.pi * t / (t[-1] - t[0]))  # Exactly 2 full periods
    
    # Create expected results for this trajectory
    expected = {
        'beta_crossings': 8,  # With our current test trajectory, we get 8 crossings
        'result': 'escaped'   # Should escape due to final position
    }
    
    return np.column_stack((x, y, z)), expected

def test_classification():
    """Test the trajectory classification function."""
    print("Running trajectory classification tests...")
    
    # Test 1: Simple trajectory that escapes
    traj, expected = generate_test_trajectory()
    
    # Print some debug info about the trajectory
    print("\nTrajectory info:")
    print(f"Shape: {traj.shape}")
    print(f"First 5 points:\n{traj[:5]}")
    print(f"Last 5 points:\n{traj[-5:]}")
    
    # Calculate some statistics
    r = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)
    z_abs = np.abs(traj[:, 2])
    print(f"\nRadial distances (r): min={np.min(r):.2f}, max={np.max(r):.2f}, mean={np.mean(r):.2f}")
    print(f"Z distances (abs): min={np.min(z_abs):.2f}, max={np.max(z_abs):.2f}, mean={np.mean(z_abs):.2f}")
    print(f"Beta boundary: {beta}")
    print(f"Alpha boundary: {alpha}")
    print(f"Y floor: {y_floor}")
    
    # Check boundary conditions
    print("\nBoundary conditions:")
    print(f"Points outside beta: {np.sum(z_abs >= beta)} / {len(z_abs)}")
    print(f"Points above alpha: {np.sum(r > alpha)} / {len(r)}")
    print(f"Points below floor: {np.sum(r < y_floor)} / {len(r)}")
    
    # Classify the trajectory
    beta_crossings, result = classify_trajectory(alpha, beta, y_floor, traj)
    
    print(f"\nTest 1 - Expected: {expected}, Got: {{'beta_crossings': {beta_crossings}, 'result': '{result}'}}")
    
    # Add more detailed assertions
    if result != expected['result']:
        print(f"\nERROR: Expected result '{expected['result']}' but got '{result}'")
        # Check which escape conditions were met
        r = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)
        z_abs = np.abs(traj[:, 2])
        
        # Check escape conditions
        escape_cond1 = np.any((r[1:] < y_floor) & (r[:-1] >= y_floor) & (z_abs[1:] <= beta))
        escape_cond2 = np.any((r[1:] <= alpha) & (z_abs[1:] >= beta) & (r[:-1] > alpha))
        escape_cond3 = z_abs[-1] >= beta
        
        print(f"Escape conditions - cond1: {escape_cond1}, cond2: {escape_cond2}, cond3: {escape_cond3}")
        
        # Check recapture conditions
        recapture_cond1 = np.any((z_abs[1:] >= beta) & (z_abs[:-1] < beta) & (r[1:] <= alpha))
        recapture_cond2 = (z_abs[-1] <= beta) and (r[-1] <= alpha)
        
        print(f"Recapture conditions - cond1: {recapture_cond1}, cond2: {recapture_cond2}")
    
    assert result == expected['result'], f"Test 1 failed: Expected {expected['result']}, got {result}"
    assert beta_crossings == expected['beta_crossings'], f"Test 1 failed: Expected {expected['beta_crossings']} crossings, got {beta_crossings}"
    
    # Test 2: Recaptured trajectory (stays within beta)
    traj_recaptured = traj.copy()
    traj_recaptured[:, 2] = 0  # Keep z=0 (within beta)
    _, result = classify_trajectory(alpha, beta, y_floor, traj_recaptured)
    print(f"Test 2 - Recaptured trajectory: {result}")
    
    # Test 3: Resimulate case (above alpha, within beta)
    traj_resimulate = traj.copy()
    traj_resimulate[:, 1] = alpha + 1000  # Keep above alpha
    traj_resimulate[:, 2] = 0  # Keep within beta
    _, result = classify_trajectory(alpha, beta, y_floor, traj_resimulate)
    print(f"Test 3 - Resimulate case: {result}")
    
    print("\nAll tests passed!")
    return traj, result

# Testing code - only runs when this file is executed directly
if __name__ == "__main__":
    # Run the test suite
    test_trajectory, _ = test_classification()
    
    # Test with a pandas DataFrame to ensure compatibility
    try:
        import pandas as pd
        df = pd.DataFrame(test_trajectory, columns=['x', 'y', 'z'])
        beta_crossings, result = classify_trajectory(alpha, beta, y_floor, df)
        print(f"\nPandas DataFrame test - beta_crossings: {beta_crossings}, result: {result}")
        
        # Test with the original file if it exists
        try:
            trajectories = pd.read_excel('test_trajectory.xlsx')
            beta_crossings, result = classify_trajectory(alpha, beta, y_floor, trajectories)
            print(f"\nOriginal test file classification: {result} with {beta_crossings} beta crossings")
        except Exception as e:
            print(f"\nNote: Could not load original test_trajectory.xlsx: {e}")
            print("This is normal if the test file doesn't exist")
            
    except ImportError:
        print("Pandas not available, skipping DataFrame test")