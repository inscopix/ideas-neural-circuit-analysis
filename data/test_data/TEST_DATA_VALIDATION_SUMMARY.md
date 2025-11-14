# Test Data Validation Summary

**Date**: 2025-11-05  
**Task**: Ensure test data for `combine_compare_state_epoch_data` is consistent with `state_epoch_baseline_analysis` output and handles sparse data scenarios

## Validation Results

### ✅ All Validation Checks Passed

All test data files have been validated and confirmed to be:
1. Consistent with `state_epoch_baseline_analysis` output format
2. Properly structured for `combine_compare_state_epoch_data` input
3. Include comprehensive sparse data test scenarios

## Test Data Inventory

### Standard Complete Data (6 subjects)
- `dummy_group1_subject1/` - Group 1, Subject 1 (complete trace + event data)
- `dummy_group1_subject2/` - Group 1, Subject 2 (complete trace + event data)
- `dummy_group1_subject3/` - Group 1, Subject 3 (complete trace + event data)
- `dummy_group2_subject1/` - Group 2, Subject 1 (complete trace + event data)
- `dummy_group2_subject2/` - Group 2, Subject 2 (complete trace + event data)
- `dummy_group2_subject3/` - Group 2, Subject 3 (complete trace + event data)

**Characteristics**:
- 150 cells per subject
- 3 states (rest, exploration, feeding) × 3 epochs (baseline, training, test) = 9 combinations
- All trace and event columns present
- Baseline: rest-baseline

### Sparse Data Scenarios (2 subjects)

#### 1. No Event Data Scenario
**Subject**: `dummy_group1_subject1_no_events/`

**Purpose**: Test handling of recordings without event detection

**Data Structure**:
- Activity CSV: Has both `mean_trace_activity` and `mean_event_rate`
- Correlation CSV: **Missing all event correlation columns**
  - Has: `max/min/mean_trace_correlation`, `positive/negative_trace_correlation`
  - Missing: All `*_event_correlation` columns
- Modulation CSV: **Missing all event modulation columns**
  - Has: `trace_modulation_scores/p_values/modulation in *`
  - Missing: All `event_modulation_*` columns

**Use Cases**:
- Testing `measure_source="trace"` with missing event data
- Mixed groups where some subjects lack event detection
- Legacy data compatibility

#### 2. Partial Event Data Scenario
**Subject**: `dummy_group1_subject2_partial_events/`

**Purpose**: Test handling of failed/incomplete event detection

**Data Structure**:
- Activity CSV: Complete data
- Correlation CSV: Event columns exist but **contain NaN for 300 rows**
  - Affected combinations: rest-baseline, exploration-test
  - Simulates failed event detection for specific state-epoch combinations
- Modulation CSV: Event columns exist but **contain NaN for specific comparisons**
  - Affected: rest-training, exploration-test comparisons

**Use Cases**:
- Robustness to incomplete event data
- Statistical analysis with missing values
- Visualization with sparse data points

## File Structure Validation

### Activity CSV (`activity_per_state_epoch_data.csv`)
- **Rows**: 1,350 (150 cells × 9 combinations)
- **Required columns**: ✅
  - `name`, `cell_index`, `state`, `epoch`
  - `mean_trace_activity`, `std_trace_activity`, `median_trace_activity`, `trace_activity_cv`
  - `mean_event_rate`

### Correlation CSV (`correlations_per_state_epoch_data.csv`)
- **Rows**: 1,350 (150 cells × 9 combinations)
- **Required trace columns**: ✅
  - `name`, `cell_index`, `state`, `epoch`
  - `max_trace_correlation`, `min_trace_correlation`, `mean_trace_correlation`
  - `positive_trace_correlation`, `negative_trace_correlation`
- **Optional event columns**: ✅ (present in standard data, missing/partial in sparse scenarios)
  - `max_event_correlation`, `min_event_correlation`, `mean_event_correlation`
  - `positive_event_correlation`, `negative_event_correlation`

### Modulation CSV (`modulation_vs_baseline_data.csv`)
- **Rows**: 1,200 (150 cells × 8 non-baseline combinations)
- **Required columns**: ✅
  - `name`, `cell_index`, `state`, `epoch`
  - `baseline_state`, `baseline_epoch`
- **Dynamic trace columns**: ✅ (16 columns for 8 comparisons)
  - `trace_modulation_scores in {state}-{epoch}`
  - `trace_p_values in {state}-{epoch}`
  - `trace_modulation in {state}-{epoch}`
- **Dynamic event columns**: ✅ (present in standard, missing/partial in sparse)
  - `event_modulation_scores in {state}-{epoch}`
  - `event_p_values in {state}-{epoch}`
  - `event_modulation in {state}-{epoch}`

## Cross-File Consistency Checks

✅ Cell counts match across all files within each subject  
✅ State-epoch combinations consistent between activity and correlation files  
✅ Baseline state/epoch consistent in modulation files  
✅ Cell indices properly numbered (0-149)  
✅ No duplicate rows within files

## Sparse Data Validation

### No Events Scenario
✅ Event correlation columns properly removed from correlation CSV  
✅ Event modulation columns properly removed from modulation CSV  
✅ Activity CSV retains `mean_event_rate` column  
✅ All trace data remains intact

### Partial Events Scenario
✅ Event correlation columns exist but contain NaN values  
✅ NaN values present for 300 rows (2 combinations × 150 cells)  
✅ Event modulation columns exist but contain NaN for specific comparisons  
✅ Trace data unaffected by partial event data

## Tools and Scripts

### Validation Script
**File**: `validate_test_data.py`

**Features**:
- Automatic detection of all subject directories
- Comprehensive structure validation
- Row count verification
- Column presence checks
- Sparse data scenario detection
- Cross-file consistency validation
- Clear success/error/warning reporting

**Usage**:
```bash
cd data/test_data
python validate_test_data.py
```

### Sparse Data Generation Script
**File**: `create_sparse_data_scenarios.py`

**Features**:
- Generates no-events scenario from complete data
- Generates partial-events scenario with strategic NaN placement
- Preserves trace data integrity
- Creates accompanying documentation

**Usage**:
```bash
cd data/test_data
python create_sparse_data_scenarios.py
```

## Expected Behavior of combine_compare_state_epoch_data

The tool should handle all test data scenarios correctly:

### With Standard Complete Data
- ✅ Load all trace and event columns
- ✅ Analyze both trace and event measures when `measure_source="both"`
- ✅ Generate separate trace/event statistical files
- ✅ Create per-group previews for both data types

### With No Events Data
- ✅ Load successfully despite missing event columns
- ✅ Use only trace data for analysis
- ✅ Generate trace-only statistical files
- ✅ Skip event previews gracefully
- ✅ Log appropriate warnings about missing event data

### With Partial Events Data
- ✅ Load successfully with NaN event values
- ✅ Skip combinations with insufficient event data
- ✅ Include valid event data in statistical analysis
- ✅ Handle NaN values in visualization
- ✅ Report which combinations were analyzed

### With Mixed Groups
- ✅ Combine subjects with different event data availability
- ✅ Use intersection of available measures
- ✅ Provide clear logging about data availability
- ✅ Generate outputs based on common measures

## Consistency with state_epoch_baseline_analysis

All test data files match the exact output format from `state_epoch_baseline_analysis`:

1. **Column naming**: Uses `trace_` and `event_` prefixes consistently
2. **Data structure**: Separate rows for each cell-state-epoch combination
3. **Modulation format**: Dynamic columns for each non-baseline comparison
4. **Baseline handling**: Baseline state/epoch stored in modulation CSV
5. **Correlation types**: Both per-cell (max/min/mean) and population (positive/negative) measures
6. **Sparse data**: Matches real-world scenarios where event detection may be incomplete

## Testing Recommendations

### Unit Tests
Test the tool with:
1. Standard complete data (all 6 subjects)
2. No events scenario (subject1_no_events)
3. Partial events scenario (subject2_partial_events)
4. Mixed group (standard + sparse subjects)

### Integration Tests
Verify:
1. `measure_source="trace"` with no events data
2. `measure_source="event"` with no events data (should gracefully handle)
3. `measure_source="both"` with mixed data availability
4. Statistical comparisons skip invalid combinations
5. Output CSVs maintain consistent structure

### Edge Cases
Test:
1. All subjects missing event data (trace-only analysis)
2. Single subject with events, rest without
3. Baseline combination handling in sparse scenarios
4. NaN propagation in statistical calculations

## Validation Command

To revalidate all test data:

```bash
cd /Users/qingliu/ideas-toolbox-epochs/data/test_data
python validate_test_data.py
```

Expected output: `✅ All validation checks passed!`

## Documentation References

- **Main README**: `data/test_data/README.md` - Overview and usage
- **Sparse Data Guide**: `data/test_data/SPARSE_DATA_README.md` - Detailed sparse data documentation
- **This Summary**: `data/test_data/TEST_DATA_VALIDATION_SUMMARY.md` - Validation results

## Conclusion

✅ **All test data is properly structured and validated**  
✅ **Consistent with state_epoch_baseline_analysis output format**  
✅ **Comprehensive sparse data scenarios included**  
✅ **Ready for use in combine_compare_state_epoch_data testing**

The test data now provides robust coverage for:
- Standard complete analysis scenarios
- Sparse/missing event data handling
- Mixed group compositions
- Real-world edge cases

This ensures the `combine_compare_state_epoch_data` tool can handle production data reliably.

