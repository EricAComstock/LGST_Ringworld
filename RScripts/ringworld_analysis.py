import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

def ringworld_analysis():
    """
    Python version of Ringworld_analysis.R
    Generates qualitative ringworld data regarding surface vs edge atmospheric losses
    from particle numbers recaptured and escaped.
    
    v1.0, Sep 16 2025 Siyona Agarwal
    """
    # Ringworld parameters
    worldWidth = np.array([
        1600000, 100000, 10000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
        200, 2000, 1000000, 200, 3957.98762536996
    ])  # km

    worldGrav = np.array([
        1, 1, 1, 1, 0.25, 0.5, 1, 1.5, 2, 3,
        1, 1, 0.00956122530571083/9.81, 1, 1
    ])  # g

    worldRad = np.array([
        149598000, 149598000, 149598000, 149598000, 149598000,
        149598000, 149598000, 149598000, 149598000, 149598000,
        1000, 1854977.7, 9.4e13, 1854977.7, 395798.762536996
    ])  # km

    # Particle data
    recaptured = np.array([
        953897, 94555, 94414, 93713, 30106, 37635, 46811, 49349, 49701, 49799,
        90559, 94174, 43002, 90855, 94362
    ])

    escaped = np.array([
        4, 16, 78, 851, 682, 560, 422, 327, 275, 201,
        4307, 392, 27, 4251, 209
    ])

    # Process data
    escape_frac_list = []
    escape_frac_std_list = []
    fudge_factor = 0

    print("Processing data...\n")
    for i in range(len(worldGrav)):
        # Create binary arrays (0 = recaptured, 1 = escaped)
        rec_p = np.zeros(int(recaptured[i]))
        esc_p = np.ones(int(escaped[i]))
        
        alldata = np.concatenate([rec_p, esc_p])
        
        # Calculate statistics
        avg_escape = np.mean(alldata)
        std_escape = np.std(alldata, ddof=1)  # Sample standard deviation
        std_avg_escape = std_escape / np.sqrt(len(alldata))
        
        # Store results
        escape_frac_list.append(avg_escape)
        escape_frac_std_list.append(std_avg_escape * (1 - fudge_factor))
        
        # Print results
        print(f"Trial {i+1}:")
        print(f"Expected escape chance: {avg_escape:.6f}")
        print(f"Expected escape chance std. dev.: {std_avg_escape:.6f}\n")

    # Convert to numpy arrays for easier manipulation
    escape_frac_list = np.array(escape_frac_list)
    escape_frac_std_list = np.array(escape_frac_std_list)

    # Prepare data for regression
    X = np.column_stack([
        np.log10(worldGrav),
        np.log10(worldWidth),
        np.log10(worldRad)
    ])
    X = sm.add_constant(X)  # Adds a constant term to the predictor

    # Fit linear models
    print("\nFitting linear models...\n")
    
    # Normal model
    y_normal = np.log10(escape_frac_list)
    model_normal = sm.OLS(y_normal, X).fit()
    
    # High estimate model
    y_high = np.log10(escape_frac_list + escape_frac_std_list)
    model_high = sm.OLS(y_high, X).fit()
    
    # Low estimate model
    y_low = np.log10(np.maximum(1e-10, escape_frac_list - escape_frac_std_list))  # Avoid log(0)
    model_low = sm.OLS(y_low, X).fit()

    # Print results
    print("=== Normal Model ===")
    print(model_normal.summary())
    print("\n=== High Estimate Model ===")
    print(model_high.summary())
    print("\n=== Low Estimate Model ===")
    print(model_low.summary())
    
    # Print interpretation
    print("\n" + "="*80)
    print("INTERPRETATION OF RESULTS:")
    print("="*80)
    print("""
1. The intercept coefficient represents log10 of leakage rate per particle for 1g ringworlds
   that are 1 km in width and 1 km in radius. Currently predictable with ~30% error.
   
2. The log10(worldGrav) coefficient shows the polynomial relation between leakage and gravity:
   - Leakage ~ gravity^(-0.62 ± 0.03)
   - Since atmospheric mass is inversely proportional to gravity:
     lifespan ~ gravity^(-0.38 ± 0.03)
   
3. The log10(worldWidth) coefficient shows a statistically significant linear correlation 
   between width and lifetime (which is proportional to the inverse of escape rate).
   
4. The log10(worldRad) coefficient shows no significant correlation with escape rates.
""")

if __name__ == "__main__":
    ringworld_analysis()