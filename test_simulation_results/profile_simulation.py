"""
Profile the particle simulation to identify performance bottlenecks.
"""
import cProfile
import pstats
import io
from pstats import SortKey
from run_simulation import run_simulation_with_params

def run_profiling():
    print("Starting profiling for 100 particles...")
    
    # Run the simulation with profiling
    pr = cProfile.Profile()
    pr.enable()
    
    try:
        # Run the simulation with 100 particles
        results = run_simulation_with_params()
        print("\nSimulation completed successfully!")
        print(f"Particles recaptured: {results.get('recaptured', 'N/A')}")
        print(f"Particles escaped: {results.get('escaped', 'N/A')}")
        escape_frac = results.get('escape_fraction')
        print(f"Escape fraction: {escape_frac*100:.4f}%" if escape_frac is not None else "Escape fraction: N/A")
    except Exception as e:
        print(f"\nSimulation failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        pr.disable()
        
        # Create a stream for the stats
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
        
        # Print the profile stats
        print("\n" + "="*80)
        print("PROFILING RESULTS (by cumulative time)")
        print("="*80)
        ps.print_stats(20)  # Show top 20 time-consuming functions
        
        # Save detailed profile to a file
        with open('simulation_profile.txt', 'w') as f:
            ps = pstats.Stats(pr, stream=f)
            ps.sort_stats(SortKey.CUMULATIVE)
            ps.print_stats()
        
        print("\nDetailed profile saved to 'simulation_profile.txt'")

if __name__ == "__main__":
    run_profiling()
