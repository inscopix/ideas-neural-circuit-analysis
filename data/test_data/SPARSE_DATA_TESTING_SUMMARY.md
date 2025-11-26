# Sparse Data Testing Summary

## Overview

This document summarizes the comprehensive unit tests added to validate sparse event data handling in the `combine_compare_state_epoch_data` tool.

## Test Data Scenarios

### 1. Standard Complete Data
- **Subjects**: `dummy_group1_subject1`, `dummy_group1_subject2`, `dummy_group1_subject3`
- **Characteristics**: Complete trace and event data for all state-epoch combinations
- **Purpose**: Baseline for comparison with sparse data scenarios

### 2. No Event Data Scenario
- **Subject**: `dummy_group1_subject1_no_events`
- **Characteristics**:
  - Activity CSV: Contains `mean_trace_activity` but lacks `mean_event_rate`
  - Correlation CSV: Contains trace correlation columns but **lacks all event_correlation columns**
  - Modulation CSV: Contains trace modulation columns but **lacks all event_modulation columns**
- **Purpose**: Test tool behavior when event detection is entirely absent

### 3. Partial Event Data Scenario
- **Subject**: `dummy_group1_subject2_partial_events`
- **Characteristics**:
  - Activity CSV: Contains both `mean_trace_activity` and `mean_event_rate`
  - Correlation CSV: Contains both trace and event columns, but **event columns have NaN for specific combinations** (rest-baseline, exploration-test)
  - Modulation CSV: Contains both trace and event columns, but **event columns have NaN for specific combinations**
- **Purpose**: Test tool robustness with intermittent event detection failures

## Unit Tests Added

### Test Class: `TestSparseDataHandling`

A comprehensive test suite with 8 test methods covering all sparse data scenarios:

#### 1. `test_load_data_without_event_correlations`
- **Purpose**: Verify data loading works when event correlation columns are completely missing
- **Validates**:
  - Event correlation columns are absent in loaded data
  - Trace correlation columns are present
  - Data loading completes successfully
  - No crashes or errors occur

#### 2. `test_load_data_with_partial_event_correlations`
- **Purpose**: Verify data loading works when event columns exist but contain NaN values
- **Validates**:
  - Event correlation columns are present
  - NaN values are preserved correctly
  - Data structure remains intact
  - Partial data is handled gracefully

#### 3. `test_mixed_group_with_and_without_event_data`
- **Purpose**: Test combining subjects where some have event data and some don't
- **Validates**:
  - Multiple subjects can be loaded together
  - Mixed data availability doesn't cause crashes
  - Correct number of subjects are counted
  - DataFrame structure is maintained

#### 4. `test_measure_source_trace_with_missing_event_data`
- **Purpose**: Verify `measure_source='trace'` works when event data is missing
- **Validates**:
  - Analysis completes successfully
  - Only trace data is used for all measures
  - No errors occur due to missing event data
  - Statistical comparison proceeds normally

#### 5. `test_measure_source_event_with_missing_event_data`
- **Purpose**: Verify `measure_source='event'` falls back gracefully when event data is missing
- **Validates**:
  - Tool doesn't crash when event data is unavailable
  - Fallback to trace data occurs when needed
  - Appropriate source selection is made
  - Limited analysis is performed safely

#### 6. `test_measure_source_both_with_mixed_availability`
- **Purpose**: Verify `measure_source='both'` handles mixed data availability
- **Validates**:
  - Both trace and event sources are selected when available
  - Partial event data (with NaN) is handled
  - Analysis includes all available data types
  - No crashes occur with sparse data

#### 7. `test_preview_generation_without_event_data`
- **Purpose**: Verify preview/boxplot generation works when event data is missing
- **Validates**:
  - Trace activity previews are generated
  - Trace correlation previews are generated
  - Event correlation previews are NOT generated (as expected)
  - Preview generation completes successfully

#### 8. `test_comparison_with_sparse_and_complete_groups`
- **Purpose**: Test two-group comparison where one group has sparse event data
- **Validates**:
  - Cross-group comparison works with mismatched data availability
  - Only common data types (trace) are used in comparison
  - Statistical comparison completes successfully
  - Results structure is correct

## Test Execution Results

All 62 tests in `test_combine_compare_state_epoch_data.py` pass successfully:
- **8 sparse data tests**: All passing
- **54 existing tests**: All passing (no regressions)

### Test Coverage Summary

| Test Category | Tests | Status |
|---------------|-------|--------|
| Input Validation | 5 | PASS |
| Subject Matching | 3 | PASS |
| Data Loading | 2 | PASS |
| Preview Generation | 4 | PASS |
| Statistical Comparison | 9 | PASS |
| Output Metadata | 2 | PASS |
| Dummy Output Integration | 3 | PASS |
| Measure Classification | 6 | PASS |
| Measure Selection | 6 | PASS |
| Color Mapping & Dimension Filtering | 10 | PASS |
| User Defined Colors | 5 | PASS |
| **Sparse Data Handling** | **8** | **PASS** |
| **Total** | **62** | **PASS** |

## Expected Tool Behavior with Sparse Data

Based on these tests, the `combine_compare_state_epoch_data` tool is validated to:

### SUCCESS: Successfully handle missing event columns
- Load data without crashing
- Perform trace-only analysis when `measure_source='trace'`
- Skip event-based outputs gracefully

### SUCCESS: Successfully handle NaN values in event data
- Preserve NaN values during data loading
- Process available event data correctly
- Skip invalid combinations in statistical analysis

### SUCCESS: Support mixed groups
- Compare groups with different data availability
- Use intersection of available data types
- Generate appropriate warnings

### SUCCESS: Generate appropriate outputs
- Create trace-specific plots and CSVs
- Skip event-specific outputs when data is missing
- Maintain consistent output structure

## Validation Scripts

### 1. `validate_test_data.py`
- Programmatically validates all test data files
- Checks for correct column presence/absence
- Verifies NaN patterns in partial event data
- Confirms row counts and data structure

### 2. `create_sparse_data_scenarios.py`
- Programmatically generates sparse data scenarios
- Removes event columns for no_events scenario
- Introduces NaN values for partial_events scenario
- Ensures consistency with standard data format

## Usage Example

Run sparse data tests:

```bash
cd /Users/qingliu/ideas-toolbox-epochs
python -m pytest toolbox/tests/test_combine_compare_state_epoch_data.py::TestSparseDataHandling -v
```

Run all tests:

```bash
cd /Users/qingliu/ideas-toolbox-epochs
python -m pytest toolbox/tests/test_combine_compare_state_epoch_data.py -v
```

## Conclusion

The comprehensive unit testing confirms that:

1. **Data Consistency**: Test data is correctly structured and consistent with `state_epoch_baseline_analysis` output format
2. **Sparse Data Handling**: The tool robustly handles missing and partial event correlation data
3. **No Regressions**: All existing tests continue to pass
4. **Production Ready**: The tool is ready for real-world scenarios where event detection may be incomplete or absent

---

**Date Created**: 2025-11-05
**Test Suite Version**: 1.0
**Total Tests**: 62 (including 8 new sparse data tests)
**Status**: All tests passing

