# Ringworld Atmospheric Loss Analysis

This R script analyzes atmospheric escape rates for ringworlds based on simulation data. It processes particle escape data to understand how different ringworld parameters affect atmospheric retention.

## Overview

The script processes simulation data from ringworld models with varying parameters to analyze atmospheric escape rates. It calculates escape fractions and performs statistical analysis to determine relationships between ringworld characteristics and atmospheric retention.

## Key Parameters Analyzed

- **Ringworld Width (km)**: Ranging from 200 to 1,600,000 km
- **Surface Gravity (g)**: Ranging from ~0.001g to 3g (Earth gravity equivalents)
- **Ringworld Radius (km)**: Ranging from 1,000 km to 9.4×10¹³ km

## Key Findings

1. **Gravity Relationship**: 
   - Leakage is proportional to gravity^(-0.62 ± 0.03)
   - Atmospheric lifespan is proportional to gravity^(-0.38 ± 0.03)

2. **Width Relationship**:
   - Statistically significant linear correlation between ringworld width and atmospheric lifetime
   - Lifetime is proportional to the inverse of escape rate

3. **Radius Impact**:
   - No significant correlation found between ringworld radius and escape rates

## Usage

1. Input data should be formatted in the script with the following vectors:
   - `worldWidth`: Ringworld widths in km
   - `worldGrav`: Surface gravity in g (Earth gravity equivalents)
   - `worldRad`: Ringworld radius in km
   - `recaptured`: Number of particles that remained in the atmosphere
   - `escaped`: Number of particles that escaped the atmosphere

2. The script will output:
   - Escape fractions for each test case
   - Standard deviations of escape fractions
   - Linear regression models for the relationships between parameters

## Dependencies

- R (tested with version 4.x)
- Base R packages (no additional packages required)

## Version History

- **V2.0.1** (2025-07-16): Current version
- **V2.0** (2025-07-12): Major update
- **V1.2.1** (2025-06-09): Bug fixes
- **V1.2** (2025-06-09): Added new features
- **V1.1** (2025-06-02): Initial improvements
- **V1.0** (2025-06-02): Initial release

## Author

Eric Comstock

## Notes

- The script includes a fudge factor (currently set to 0) that can be adjusted if needed
- Results are printed to the console and can be redirected to a file if desired
