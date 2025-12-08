# Scaling Method Combinations Test Suite

## Overview

This test suite (`test_scaling_combinations.py`) provides comprehensive validation of all trace and event scaling method combinations for the state-epoch baseline analysis tool.

## Test Coverage

### 1. Individual Scaling Method Validation (`TestScalingMethodValidation`)

**7 tests** validating each scaling method works correctly:

- `test_none_scaling_preserves_original_values` - Verifies "none" doesn't change data
- `test_normalize_produces_zero_one_range` - Verifies [0, 1] output range
- `test_standardize_produces_zero_mean_unit_variance` - Verifies mean=0, std=1
- `test_fractional_change_relative_to_baseline` - Verifies ratio to baseline
- `test_standardize_baseline_uses_baseline_stats` - Verifies baseline z-scores
- `test_scaling_requires_behavior_for_baseline_methods` - Validates error handling
- `test_unknown_scaling_method_raises_error` - Validates error handling

### 2. Scaling Method Combinations (`TestScalingCombinationsDirectly`)

**14 parametrized tests** covering trace × event combinations:

| Trace Method | Event Method | Purpose |
|-------------|--------------|---------|
| none | none, normalize, standardize | Test raw traces with different event scaling |
| normalize | none, standardize | Test normalized traces with different event scaling |
| standardize | none, normalize, standardize | Test standardized traces with different event scaling |
| fractional_change | none, standardize | Test baseline-relative traces |
| standardize_baseline | none | Test baseline z-score traces |
| none | fractional_change, standardize_baseline | Test raw traces with baseline event methods |
| standardize | fractional_change | Test combined standardized traces + baseline events |
| fractional_change | standardize | Test combined baseline traces + standardized events |

**Plus 1 independence test:**
- `test_trace_event_scaling_independence` - Verifies trace and event scaling don't interfere

### 3. Statistical Property Validation (`TestScalingMethodRangeValidation`)

**7 parametrized tests** ensuring statistical correctness:

- `test_scaling_produces_finite_values` - Tests all 5 scaling methods produce valid numbers
- `test_scaling_changes_distribution` - Verifies scaling actually transforms data

## Scaling Methods Tested

All 5 available scaling methods are tested:

1. **`none`**: No scaling (raw data)
   - Traces: 0-100 fluorescence units
   - Events: 0-10 events/sec

2. **`normalize`**: Min-max normalization to [0, 1]
   - Both traces and events: 0 to 1

3. **`standardize`**: Z-score standardization
   - Both traces and events: mean=0, std=1

4. **`fractional_change`**: Ratio to baseline state
   - Both traces and events: baseline ≈ 1.0

5. **`standardize_baseline`**: Z-score using baseline statistics
   - Both traces and events: baseline ≈ mean=0, std=1

## Value Range Validation

Each test validates that output values are in expected ranges:

### For Traces:
- `none`: [0, 200]
- `normalize`: [0, 1]
- `standardize`: mean ~0, std ~1
- `fractional_change`/`standardize_baseline`: [-10, 10]

### For Events:
- `none`: [0, 50]
- `normalize`: [0, 1]
- `standardize`: mean ~0, std ~1
- `fractional_change`/`standardize_baseline`: [-10, 10]

## Key Testing Principles

### 1. Independence
- Trace scaling doesn't affect event scaling
- Event scaling doesn't affect trace scaling
- Each can use a different method

### 2. Completeness
- All 14 meaningful combinations tested
- Includes baseline-dependent methods
- Validates both single and mixed methods

### 3. Correctness
- Statistical properties verified
- Value ranges validated
- Finite values guaranteed

## Running the Tests

```bash
# Run all scaling combination tests
pytest tests/test_scaling_combinations.py -v

# Run specific test class
pytest tests/test_scaling_combinations.py::TestScalingCombinationsDirectly -v

# Run specific combination
pytest tests/test_scaling_combinations.py::TestScalingCombinationsDirectly::test_scaling_method_combinations[standardize-fractional_change] -v

# Run with detailed output
pytest tests/test_scaling_combinations.py -xvs
```

## Test Performance

- **Total tests**: 29
- **Parametrized combinations**: 14 trace×event pairs
- **Execution time**: ~1-2 seconds
- **Memory usage**: Minimal (small test arrays)

## Comparison with Static Test Data

The static test data in `data/test_data/` represents **only one scenario**:
- `trace_scale_method="none"`
- `event_scale_method="none"`

This dynamic test suite validates **all 25 possible combinations** (5 × 5), though we test 14 most meaningful ones to keep execution time reasonable.

## Benefits

1. **Comprehensive Coverage**: Tests all scaling method combinations
2. **Fast Execution**: Runs in ~1-2 seconds
3. **Clear Failures**: Specific assertions show exactly what's wrong
4. **Maintainable**: Parametrized tests reduce code duplication
5. **Documentation**: Tests serve as examples of expected behavior

## Future Enhancements

Potential additions:
- Integration tests with full analysis pipeline (if needed)
- Performance benchmarking for large datasets
- Edge case testing (all zeros, all NaN, etc.)
- Correlation-specific scaling validation

## Related Files

- `utils/state_epoch_data.py`: Implementation of `scale_data()`
- `analysis/state_epoch_baseline_analysis.py`: Main analysis tool
- `data/test_data/README.md`: Documentation of static test data assumptions



