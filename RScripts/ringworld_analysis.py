"""
File: ringworld_analysis.py
Author: Siyona Agarwal <siyona.agarwal@example.com>
Date: 2025-09-16
Description: Python implementation of Ringworld_analysis.R. This script generates
             qualitative ringworld data regarding surface vs edge atmospheric losses
             from particle numbers recaptured and escaped.

Version History:
    v1.0 (2025-09-16): Initial implementation
    v1.1 (2025-09-22): Updated to follow coding style guide
"""

import numpy as np
import statsmodels.api as sm
from typing import Tuple, Dict, Any

# Constants
FUDGE_FACTOR = 0.0
MIN_LOG_VALUE = 1e-10  # Small value to avoid log(0)


def calculate_escape_statistics(
    recaptured: np.ndarray, escaped: np.ndarray, fudge: float = FUDGE_FACTOR
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate escape statistics from recaptured and escaped particle counts.

    Args:
        recaptured: Array of counts of recaptured particles
        escaped: Array of counts of escaped particles
        fudge: Fudge factor for standard deviation adjustment

    Returns:
        Tuple containing:
            - Array of escape fractions
            - Array of standard deviations of escape fractions
    """
    escape_frac_list = []
    escape_frac_std_list = []

    for rec, esc in zip(recaptured, escaped):
        # Create binary arrays (0 = recaptured, 1 = escaped)
        rec_p = np.zeros(int(rec))
        esc_p = np.ones(int(esc))
        all_data = np.concatenate([rec_p, esc_p])

        # Calculate statistics
        avg_escape = np.mean(all_data)
        std_escape = np.std(all_data, ddof=1)  # Sample standard deviation
        std_avg_escape = std_escape / np.sqrt(len(all_data))

        # Store results
        escape_frac_list.append(avg_escape)
        escape_frac_std_list.append(std_avg_escape * (1 - fudge))

        # Print trial results
        print(f"Trial {len(escape_frac_list)}:")
        print(f"Expected escape chance: {avg_escape:.6f}")
        print(f"Expected escape chance std. dev.: {std_avg_escape:.6f}\n")

    return np.array(escape_frac_list), np.array(escape_frac_std_list)


def fit_linear_models(
    world_grav: np.ndarray,
    world_width: np.ndarray,
    world_rad: np.ndarray,
    escape_frac: np.ndarray,
    escape_std: np.ndarray,
) -> Dict[str, Any]:
    """
    Fit linear models to the ringworld data.

    Args:
        world_grav: Array of gravity values (g)
        world_width: Array of ringworld widths (km)
        world_rad: Array of ringworld radii (km)
        escape_frac: Array of escape fractions
        escape_std: Array of escape fraction standard deviations

    Returns:
        Dictionary containing the fitted models
    """
    # Prepare data for regression
    X = np.column_stack(
        [np.log10(world_grav), np.log10(world_width), np.log10(world_rad)]
    )
    X = sm.add_constant(X)  # Add constant term to predictor

    # Fit models
    y_normal = np.log10(escape_frac)
    y_high = np.log10(escape_frac + escape_std)
    y_low = np.log10(np.maximum(MIN_LOG_VALUE, escape_frac - escape_std))

    return {
        "normal": sm.OLS(y_normal, X).fit(),
        "high": sm.OLS(y_high, X).fit(),
        "low": sm.OLS(y_low, X).fit(),
    }


def print_interpretation() -> None:
    """Print interpretation of the analysis results."""
    print("\n" + "=" * 80)
    print("INTERPRETATION OF RESULTS:")
    print("=" * 80)
    print(
        """
1. The intercept coefficient represents log10 of leakage rate per particle for 1g ringworlds
   that are 1 km in width and 1 km in radius. Currently predictable with ~30% error.
   
2. The log10(worldGrav) coefficient shows the polynomial relation between leakage and gravity:
   - Leakage ~ gravity^(-0.62 ± 0.03)
   - Since atmospheric mass is inversely proportional to gravity:
     lifespan ~ gravity^(-0.38 ± 0.03)
   
3. The log10(worldWidth) coefficient shows a statistically significant linear correlation 
   between width and lifetime (which is proportional to the inverse of escape rate).
   
4. The log10(worldRad) coefficient shows no significant correlation with escape rates.
"""
    )


def main() -> None:
    """Main function to run the ringworld analysis."""
    # Ringworld parameters
    world_width = np.array(
        [
            1_600_000, 100_000, 10_000, 1_000, 1_000, 1_000, 1_000, 1_000, 1_000, 1_000,
            200, 2_000, 1_000_000, 200, 3_957.98762536996,
        ]
    )  # km

    world_grav = np.array(
        [
            1, 1, 1, 1, 0.25, 0.5, 1, 1.5, 2, 3,
            1, 1, 0.00956122530571083 / 9.81, 1, 1,
        ]
    )  # g

    world_rad = np.array(
        [
            149_598_000, 149_598_000, 149_598_000, 149_598_000, 149_598_000,
            149_598_000, 149_598_000, 149_598_000, 149_598_000, 149_598_000,
            1_000, 1_854_977.7, 9.4e13, 1_854_977.7, 395_798.762536996,
        ]
    )  # km

    # Particle data
    recaptured = np.array(
        [
            95_3897, 94_555, 94_414, 93_713, 30_106, 37_635, 46_811, 49_349, 49_701, 49_799,
            90_559, 94_174, 43_002, 90_855, 94_362,
        ]
    )

    escaped = np.array(
        [
            4, 16, 78, 851, 682, 560, 422, 327, 275, 201,
            4_307, 392, 27, 4_251, 209,
        ]
    )

    # Process data
    print("Processing data...\n")
    escape_frac, escape_std = calculate_escape_statistics(recaptured, escaped)

    # Fit models
    print("\nFitting linear models...\n")
    models = fit_linear_models(world_grav, world_width, world_rad, escape_frac, escape_std)

    # Print results
    print("=== Normal Model ===")
    print(models["normal"].summary())
    print("\n=== High Estimate Model ===")
    print(models["high"].summary())
    print("\n=== Low Estimate Model ===")
    print(models["low"].summary())

    # Print interpretation
    print_interpretation()


if __name__ == "__main__":
    main()