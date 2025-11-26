# Test Data for State-Epoch Analysis

This directory contains test data for the `combine_compare_state_epoch_data` tool. The data has been updated to match the new output format from `state_epoch_baseline_analysis` which separates trace and event metrics.

## Directory Structure

```
test_data/
├── dummy_group1_subject1/          # Standard complete data
│   ├── activity_per_state_epoch_data.csv
│   ├── correlations_per_state_epoch_data.csv
│   └── modulation_vs_baseline_data.csv
├── dummy_group1_subject2/          # Standard complete data
│   └── ... (same files)
├── dummy_group1_subject3/          # Standard complete data
│   └── ... (same files)
├── dummy_group2_subject1/          # Standard complete data
│   └── ... (same files)
├── dummy_group2_subject2/          # Standard complete data
│   └── ... (same files)
├── dummy_group2_subject3/          # Standard complete data
│   └── ... (same files)
├── dummy_group1_subject1_no_events/    # Sparse data: NO event correlation data
│   └── ... (trace-only files)
├── dummy_group1_subject2_partial_events/  # Sparse data: PARTIAL event data
│   └── ... (some NaN event values)
├── SPARSE_DATA_README.md           # Documentation for sparse data scenarios
├── create_sparse_data_scenarios.py # Script to regenerate sparse data
└── validate_test_data.py           # Validation script for all test data
```

## Data Specifications

### States and Epochs
- **States**: rest, exploration, feeding
- **Epochs**: baseline, training, test
- **Baseline**: rest-baseline
- **Cells**: 150 cells per subject

### File Formats

#### 1. `activity_per_state_epoch_data.csv`

Contains trace activity and event rate data for each cell in each state-epoch combination.

**Columns:**
- `name`: Cell identifier (e.g., "cell_0")
- `cell_index`: Numeric cell index
- `state`: Behavioral state
- `epoch`: Experimental epoch
- `mean_trace_activity`: Mean calcium trace activity
- `std_trace_activity`: Standard deviation of trace activity
- `median_trace_activity`: Median trace activity
- `trace_activity_cv`: Coefficient of variation for trace activity
- `mean_event_rate`: Mean event (spike) rate per second

**Rows**: 1,350 (150 cells × 3 states × 3 epochs)

#### 2. `correlations_per_state_epoch_data.csv`

Contains both per-cell and population-level correlation statistics for trace and event data.

**Columns:**
- `name`: Cell identifier
- `cell_index`: Numeric cell index
- `state`: Behavioral state
- `epoch`: Experimental epoch

*Per-cell trace correlations:*
- `max_trace_correlation`: Maximum correlation with other cells (trace)
- `min_trace_correlation`: Minimum correlation with other cells (trace)
- `mean_trace_correlation`: Mean correlation across all cell pairs (trace)

*Per-cell event correlations:*
- `max_event_correlation`: Maximum correlation with other cells (events)
- `min_event_correlation`: Minimum correlation with other cells (events)
- `mean_event_correlation`: Mean correlation across all cell pairs (events)

*Population-level correlations:*
- `positive_trace_correlation`: Mean positive correlation (trace)
- `negative_trace_correlation`: Mean negative correlation (trace)
- `positive_event_correlation`: Mean positive correlation (events)
- `negative_event_correlation`: Mean negative correlation (events)

**Rows**: 1,350 (150 cells × 3 states × 3 epochs)

#### 3. `modulation_vs_baseline_data.csv`

Contains modulation analysis comparing each state-epoch combination to the baseline (rest-baseline).

**Columns:**
- `name`: Cell identifier
- `cell_index`: Numeric cell index
- `state`: Behavioral state being compared
- `epoch`: Experimental epoch being compared
- `baseline_state`: Baseline state ("rest")
- `baseline_epoch`: Baseline epoch ("baseline")

*Dynamic columns for each non-baseline state-epoch combination:*

For trace data:
- `trace_modulation_scores in {state}-{epoch}`: Z-score of modulation
- `trace_p_values in {state}-{epoch}`: Statistical significance
- `trace_modulation in {state}-{epoch}`: Categorical classification (1=up, 0=no change, -1=down)

For event data:
- `event_modulation_scores in {state}-{epoch}`: Z-score of modulation
- `event_p_values in {state}-{epoch}`: Statistical significance
- `event_modulation in {state}-{epoch}`: Categorical classification (1=up, 0=no change, -1=down)

**Rows**: 1,200 (150 cells × 8 non-baseline combinations)
**Total columns**: 54 (6 fixed + 48 dynamic for 8 state-epoch combinations)

## Data Characteristics

### Group Differences
- **Group 1**: Lower baseline activity and event rates
- **Group 2**: Higher baseline activity and event rates (+30% trace, +30% events)

### State Modulation
- **Rest**: Baseline activity levels
- **Exploration**: +30% activity increase
- **Feeding**: +50% activity increase

### Epoch Effects
- **Baseline**: Reference epoch
- **Training**: +20% activity increase
- **Test**: +40% activity increase

### Subject Variability
- Each subject within a group has slight variations (±5% per subject number)
- Cell-to-cell variability follows normal distribution (σ = 10% of mean)

## Updating Test Data

To regenerate the test data with consistent structure:

```bash
cd data/test_data
python update_test_data.py
```

This script ensures:
1. All column names match the current state-epoch output format
2. Trace and event data are properly separated
3. Data is consistent across all subjects
4. Realistic correlations and modulations are generated

## Scaling Method Assumptions

### Important: These test data represent a specific scaling scenario

The test data CSV files contain **pre-processed output** from `state_epoch_baseline_analysis` with the following scaling assumptions:

**Trace Scaling**: `trace_scale_method="none"` (raw fluorescence units)
- Activity values: ~1.0-2.5 arbitrary fluorescence units
- Represents unscaled calcium trace data

**Event Scaling**: `event_scale_method="none"` (raw event rates)
- Event rate values: ~5-10 events per second
- Represents unscaled detected calcium events

**Correlation Data**: Always in [-1, 1] range regardless of scaling
- Correlations are computed AFTER scaling is applied
- Both trace and event correlations follow standard correlation bounds

**Modulation Data**: Z-scores (standardized)
- Modulation scores represent standardized differences from baseline
- Separate calculations for trace and event data

### Scaling Method Impact

In real usage, traces and events can use **different scaling methods independently**:

| Method | Value Range | Description |
|--------|-------------|-------------|
| `none` | Original units | No scaling applied (what this test data uses) |
| `normalize` | [0, 1] | Min-max normalization |
| `standardize` | ~[-3, 3] | Z-score (mean=0, std=1) |
| `fractional_change` | [-1, ∞] | Percentage change from baseline state |
| `standardize_baseline` | ~[-3, 3] | Z-score using baseline state statistics |

**Example**: Real data might use:
- `trace_scale_method="standardize"` → trace activity ~0.0 ± 1.0
- `event_scale_method="fractional_change"` → event rates as % change from baseline
- This would produce very different value ranges than the test data!

### Test Data Limitations

These test data are suitable for:
- ✅ Testing CSV structure and column names
- ✅ Testing data loading and grouping functionality
- ✅ Testing statistical comparison logic
- ✅ Verifying proper separation of trace vs event metrics

These test data are NOT suitable for:
- ❌ Testing different scaling method combinations
- ❌ Validating scaling method behavior
- ❌ Testing scale-dependent statistical properties

For testing different scaling methods, use dynamically generated data with the actual analysis tools rather than these static CSV files.

## Changelog

### 2025-11-05
- **Added sparse data test scenarios** for robust testing of missing/partial event data
- Created `dummy_group1_subject1_no_events/` - Complete absence of event correlation data
- Created `dummy_group1_subject2_partial_events/` - Partial event data with NaN values
- Added `SPARSE_DATA_README.md` with detailed usage examples
- Added `create_sparse_data_scenarios.py` script to regenerate sparse data
- Added `validate_test_data.py` comprehensive validation script
- Enhanced main README with sparse data documentation
- All test data validated and consistent with `state_epoch_baseline_analysis` output format

### 2025-11-03
- **BREAKING CHANGE**: Updated column names to separate trace and event data
- Activity columns: `mean_activity` → `mean_trace_activity`, etc.
- Added event columns: `mean_event_rate` in activity, event correlations, event modulation
- Correlation columns: `max_correlation` → `max_trace_correlation`, etc.
- Added population-level positive/negative correlations for trace and event data
- Modulation columns now have separate `trace_` and `event_` prefixes
- Increased cell count to 150 cells for more realistic testing
- **Added scaling method assumptions**: Documented that test data uses `scale_method="none"`
- Updated both `data/test_data/` and `outputs/dummy_group*/` directories

### Previous Format (Deprecated)
- Used generic column names without trace/event separation
- Only 50 cells per subject
- Missing event-based metrics
- No documentation of scaling assumptions

## Sparse Data Test Scenarios

### Overview

In addition to standard complete test data, this directory includes **sparse data scenarios** to test the robustness of `combine_compare_state_epoch_data` when handling missing or partial event detection data.

### Sparse Data Subjects

1. **dummy_group1_subject1_no_events/**
   - Simulates recordings without event detection
   - Correlation CSV: Missing all event_correlation columns
   - Modulation CSV: Missing all event_modulation columns
   - Use case: Testing trace-only analysis with `measure_source="trace"`

2. **dummy_group1_subject2_partial_events/**
   - Simulates failed event detection for some combinations
   - Correlation CSV: Event columns exist but contain NaN for some state-epoch combinations
   - Modulation CSV: Event columns exist but contain NaN for some comparisons
   - Use case: Testing robustness to incomplete event data

### Testing Sparse Data Scenarios

Run validation script to check all test data consistency:

```bash
cd data/test_data
python validate_test_data.py
```

Regenerate sparse data scenarios if needed:

```bash
cd data/test_data
python create_sparse_data_scenarios.py
```

For detailed usage examples and validation checklist, see [SPARSE_DATA_README.md](SPARSE_DATA_README.md).

### Expected Behavior with Sparse Data

The `combine_compare_state_epoch_data` tool should:

- ✅ Load data successfully when event columns are missing
- ✅ Handle NaN values in event data gracefully
- ✅ Fall back to trace data when `measure_source="trace"`
- ✅ Skip invalid combinations in statistical analysis
- ✅ Generate appropriate warnings for missing data
- ✅ Support mixed groups (some subjects with events, some without)

