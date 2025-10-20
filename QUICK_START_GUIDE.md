# Ringworld Simulation Quick Start Guide

This guide shows you how to run ringworld atmospheric simulations with different configurations and particle counts.

## 🚀 Quick Start (Interactive Mode)

**Easiest way to get started:**

```bash
python run_quick_test.py
```

Then follow the prompts:
- **For beginners**: Enter `ultra single` (fastest test)
- **For thorough testing**: Enter `quick batch-3` (test 3 configurations)
- **For comprehensive analysis**: Enter `standard batch-all` (test all configurations)

## 🔧 Programmatic Usage (Custom Particle Counts)

### Import the Function
```python
from run_quick_test import run_simulation_test
```

### Basic Examples

#### Quick Tests (Fast, for testing)
```python
# Fastest possible test - Bishop Ring with 25 particles
run_simulation_test('ultra', single_config=True)

# Quick test of medium ringworld - 100 particles  
run_simulation_test('quick', priority=6)

# Test largest ringworld - 25 particles
run_simulation_test('ultra', last_row=True)
```

#### Custom Particle Counts
```python
# Small tests (10-100 particles) - Very fast
run_simulation_test('ultra', single_config=True, custom_particles=10)      # 10 particles
run_simulation_test('quick', priority=1, custom_particles=50)              # 50 particles
run_simulation_test('ultra', row_index=0, custom_particles=100)            # 100 particles

# Medium tests (500-2000 particles) - Balanced speed/accuracy
run_simulation_test('quick', priority=6, custom_particles=500)             # 500 particles
run_simulation_test('standard', last_row=True, custom_particles=1000)      # 1,000 particles
run_simulation_test('quick', row_index=5, custom_particles=2000)           # 2,000 particles

# Large tests (5000-20000 particles) - Thorough, slower
run_simulation_test('standard', priority=8, custom_particles=5000)         # 5,000 particles
run_simulation_test('long', single_config=True, custom_particles=10000)    # 10,000 particles
run_simulation_test('standard', priority=22, custom_particles=20000)       # 20,000 particles

# Production tests (50000+ particles) - Very thorough, much slower
run_simulation_test('full', priority=1, custom_particles=50000)            # 50,000 particles
run_simulation_test('full', row_index=2, custom_particles=100000)          # 100,000 particles
```

## 📊 Test Modes (Speed vs Accuracy)

| Mode     | Default Particles | Time per Sim | Use Case |
|----------|-------------------|--------------|----------|
| `ultra`    | 25                | 0.2-0.5s     | Quick testing, debugging |
| `quick`    | 100               | 1-3s         | **Recommended** for most use |
| `standard` | 1000              | 10-30s       | Thorough analysis |
| `long`     | 1000              | 1-5 min      | Very thorough analysis |
| `full`     | 1000              | 5-20 min     | Comprehensive study |

## 🎯 Selection Methods

### Select Specific Ringworlds
```python
# By priority number (see table below)
run_simulation_test('quick', priority=1)        # Bishop Ring
run_simulation_test('quick', priority=22)       # Seyfert Ringworld (medium)

# By row index (0-based)
run_simulation_test('quick', row_index=0)       # First row (Bishop Ring)
run_simulation_test('quick', row_index=13)      # Last row (Seyfert medium)

# Special selections
run_simulation_test('quick', single_config=True)  # Lowest priority (Bishop Ring)
run_simulation_test('quick', last_row=True)       # Highest priority (Seyfert medium)
```

### Batch Tests (Multiple Configurations)
```python
# Test multiple configurations
run_simulation_test('ultra', max_configs=3)     # First 3 configurations
run_simulation_test('quick', max_configs=5)     # First 5 configurations
run_simulation_test('ultra')                    # ALL configurations (14 total)
```

## 🌍 Available Ringworld Configurations

| Row | Priority | Name | Radius (km) | Notes |
|-----|----------|------|-------------|-------|
| 0 | 1 | Bishop Ring | 1,000 | Smallest, fastest to simulate |
| 1 | 4 | T5-class Ringworld | 149,598 | Earth orbit size |
| 2 | 6 | M3-class Ringworld | 18,922,800 | Medium size |
| 3 | 7 | A5-class Ringworld | 524,660,016 | Large |
| 4 | 8 | O8-class Ringworld | 61,680,782,226 | Very large |
| 5 | 10 | Large Banks Orbital | 1,854,978 | Small orbital |
| 6 | 12 | M9-class Ringworld | 2,591,111 | Small-medium |
| 7 | 13 | K4-class Ringworld | 66,902,202 | Large |
| 8 | 14 | F5-class Ringworld | 285,022,224 | Large |
| 9 | 15 | B8-class Ringworld | 1,862,478,470 | Very large |
| 10 | 16 | B1-class Ringworld | 17,375,262,996 | Extremely large |
| 11 | 17 | O3-class Ringworld | 177,006,590,000 | Ultra large |
| 12 | 21 | Seyfert Ringworld (small) | 2,115,630,000,000 | Massive |
| 13 | 22 | Seyfert Ringworld (medium) | 8,193,810,000,000 | Largest |

## 📁 Output Files

All results are saved to organized directories:

```
simulation_results/
├── single_config/
│   ├── ultra_mode/
│   ├── quick_mode/
│   └── standard_mode/
└── batch_tests/
    ├── ultra_mode/
    ├── quick_mode/
    └── standard_mode/
```

Each test creates a timestamped directory containing:
- **Particle data**: Individual trajectory results
- **Summary files**: Statistical analysis (CSV and Excel)
- **Log files**: Detailed execution logs
- **Leak rate analysis**: Atmospheric lifetime calculations

## 🎯 Recommended Usage Patterns

### For Beginners
```python
# Start with these simple tests
run_simulation_test('ultra', single_config=True)              # Fastest test
run_simulation_test('quick', priority=6, custom_particles=100) # Medium test
run_simulation_test('ultra', max_configs=3)                   # Compare 3 configs
```

### For Research
```python
# Thorough analysis of specific ringworlds
run_simulation_test('standard', priority=1, custom_particles=5000)   # Bishop Ring
run_simulation_test('standard', priority=6, custom_particles=5000)   # M3-class
run_simulation_test('long', priority=22, custom_particles=2000)      # Seyfert (slower)
```

### For Production
```python
# High-resolution simulations (warning: very slow!)
run_simulation_test('full', priority=1, custom_particles=50000)      # 50k particles
run_simulation_test('full', priority=6, custom_particles=100000)     # 100k particles
```

## 🔍 Example Script

Try running the example script for guided usage:

```bash
python example_usage.py
```

This script demonstrates various usage patterns and lets you choose what to run.

## ⚡ Performance Tips

1. **Start small**: Use `ultra` mode with 10-100 particles for testing
2. **Scale up gradually**: Increase particles only when needed
3. **Use appropriate modes**: 
   - `ultra`/`quick` for testing and debugging
   - `standard` for research
   - `long`/`full` only for final production runs
4. **Seyfert ringworlds**: These are extremely large and take longer to simulate
5. **Batch tests**: Use `max_configs=3` instead of all configs for initial exploration

## 🆘 Troubleshooting

- **"Values in t_eval are not within t_span"**: Fixed in the optimized version
- **"name 'd' is not defined"**: Fixed in leak rate calculation
- **Very long simulation times**: Use smaller particle counts or `ultra` mode
- **Memory issues**: Reduce particle count or use smaller ringworlds first

## 📞 Quick Reference

**Fastest test**: `run_simulation_test('ultra', single_config=True, custom_particles=10)`

**Recommended test**: `run_simulation_test('quick', priority=6, custom_particles=100)`

**Thorough test**: `run_simulation_test('standard', priority=1, custom_particles=1000)`

**Comparison test**: `run_simulation_test('ultra', max_configs=3)`
