# Simulation Results Directory Structure

The `run_quick_test.py` script now organizes all output files into a clean, hierarchical directory structure:

## 📁 Directory Organization

```
simulation_results/
├── single_config/                    # Single configuration tests
│   ├── ultra_mode/                   # Ultra-quick tests (25 particles)
│   ├── quick_mode/                   # Quick tests (100 particles)
│   ├── standard_mode/                # Standard tests (1000 particles)
│   ├── long_mode/                    # Long tests (1000 particles, long sim)
│   └── full_mode/                    # Full tests (1000 particles, full sim)
└── batch_tests/                      # Multi-configuration batch tests
    ├── ultra_mode/                   # Ultra-quick batch tests
    ├── quick_mode/                   # Quick batch tests
    ├── standard_mode/                # Standard batch tests
    ├── long_mode/                    # Long batch tests
    └── full_mode/                    # Full batch tests
```

## 🏷️ Naming Convention

### Single Configuration Tests
- **Default**: `{mode}_mode/{config_name}_{timestamp}/`
- **Priority**: `{mode}_mode/priority_{priority}_{config_name}_{timestamp}/`
- **Row Index**: `{mode}_mode/row_{index}_{config_name}_{timestamp}/`
- **Last Row**: `{mode}_mode/last_row_{config_name}_{timestamp}/`
- **Custom Particles**: `{mode}_mode/{config_name}_{particles}p_{timestamp}/`

### Batch Tests
- **Limited**: `{mode}_mode/batch_{count}_configs_{timestamp}/`
- **All Configs**: `{mode}_mode/all_configs_{timestamp}/`

## 📄 File Contents

Each test directory contains:

```
{test_directory}/
├── detailed_results/
│   └── {designation}_{particles}particles_siyona_{timestamp}/
│       ├── {designation}_{particles}particles_siyona_{timestamp}_metadata.json
│       ├── {designation}_{particles}particles_siyona_{timestamp}_particles.csv
│       └── {designation}_{particles}particles_siyona_{timestamp}_particles.xlsx
└── logs/
    └── parameter_sweep_{timestamp}.log
```

## 🎯 Examples

### Single Configuration Examples
- `simulation_results/single_config/ultra_mode/Bishop_Ring_20251014_203232/`
- `simulation_results/single_config/quick_mode/priority_22_Seyfert_Ringworld_medium_20251014_203245/`
- `simulation_results/single_config/standard_mode/row_13_Seyfert_Ringworld_medium_20251014_203301/`
- `simulation_results/single_config/ultra_mode/last_row_Seyfert_Ringworld_medium_20251014_203315/`

### Batch Test Examples
- `simulation_results/batch_tests/ultra_mode/batch_3_configs_20251014_203330/`
- `simulation_results/batch_tests/quick_mode/all_configs_20251014_203345/`

## 🔧 Benefits

1. **Organized**: Easy to find specific test results
2. **Timestamped**: No file conflicts, clear chronology
3. **Hierarchical**: Logical grouping by test type and mode
4. **Descriptive**: Directory names clearly indicate content
5. **Scalable**: Structure supports any number of tests
6. **Clean**: All results contained in `simulation_results/` directory

## 🚀 Usage

The new directory structure is automatically created when running tests:

```bash
# Single configuration tests
python run_quick_test.py
# Then enter: "ultra row-0" → Creates single_config/ultra_mode/row_0_Bishop_Ring_{timestamp}/

# Batch tests  
python run_quick_test.py
# Then enter: "quick batch-all" → Creates batch_tests/quick_mode/all_configs_{timestamp}/
```

All output files are automatically organized into the appropriate directories with no additional configuration required!
