# Sparse Event Data Test Scenarios

This directory contains additional test subjects with sparse event data to test
the `combine_compare_state_epoch_data` tool's ability to handle missing or
partial event detection.

## Test Scenarios

### 1. dummy_group1_subject1_no_events/
**Scenario**: Complete absence of event correlation data

**Purpose**: Test tool behavior when event correlation columns are entirely
missing from input CSV files (simulates recordings without event detection).

**Data Characteristics**:
- Activity CSV: Contains both `mean_trace_activity` and `mean_event_rate`
- Correlation CSV: Contains ONLY trace correlation columns
  - Has: max/min/mean_trace_correlation, positive/negative_trace_correlation
  - Missing: All event correlation columns
- Modulation CSV: Contains ONLY trace modulation columns
  - Missing: All event_modulation_* columns

**Expected Behavior**:
- Tool should load data successfully
- Analysis should proceed using only trace data
- No errors when event columns are missing
- measure_source="event" should gracefully handle missing data

### 2. dummy_group1_subject2_partial_events/
**Scenario**: Partial event correlation data (some NaN values)

**Purpose**: Test tool behavior when event detection succeeded for some
state-epoch combinations but failed for others.

**Data Characteristics**:
- Activity CSV: Complete data for all combinations
- Correlation CSV: Event correlation columns exist but contain NaN for:
  - rest-baseline (simulating baseline without events)
  - exploration-test (simulating failed event detection)
- Modulation CSV: Event modulation columns exist but contain NaN for:
  - rest-training comparison
  - exploration-test comparison

**Expected Behavior**:
- Tool should load data successfully
- Analysis should handle NaN values gracefully
- Statistical comparisons should skip combinations with insufficient data
- Visualization should handle sparse data appropriately

## Usage with combine_compare_state_epoch_data

### Test Case 1: Mixed Group (Some Subjects with Events, Some Without)

```bash
# Group 1: Mix of subjects with and without event data
group1_activity=(
    "dummy_group1_subject1/activity_per_state_epoch_data.csv"
    "dummy_group1_subject1_no_events/activity_per_state_epoch_data.csv"
    "dummy_group1_subject2_partial_events/activity_per_state_epoch_data.csv"
)

# This should handle sparse event data gracefully
python -m toolbox.tools.combine_compare_state_epoch_data \
    --group1_activity_csv_files "${group1_activity[@]}" \
    --group1_correlation_csv_files ... \
    --group1_modulation_csv_files ... \
    --measure_source trace \
    --output_dir output_sparse_test/
```

**Expected**: Tool should successfully analyze using trace data only.

### Test Case 2: Attempt Event Analysis with Sparse Data

```bash
# Force event analysis with sparse data
python -m toolbox.tools.combine_compare_state_epoch_data \
    --measure_source event \
    ...
```

**Expected**: Tool should either:
1. Fall back to trace data if event data missing, OR
2. Provide clear error message about insufficient event data

### Test Case 3: Both Trace and Event Analysis

```bash
# Analyze both when available
python -m toolbox.tools.combine_compare_state_epoch_data \
    --measure_source both \
    ...
```

**Expected**: Tool should analyze trace data for all subjects and event data
only where available, creating separate trace/event output files.

## Validation Checklist

When testing with sparse data, verify:

- [ ] Tool loads CSV files without errors when event columns missing
- [ ] Tool handles NaN values in event columns gracefully
- [ ] Statistical comparisons skip invalid/missing data
- [ ] Output CSVs maintain consistent structure
- [ ] Visualization handles sparse data without crashing
- [ ] measure_source parameter respects sparse data availability
- [ ] Error messages are clear when requested data type unavailable
- [ ] Mixed groups (some with events, some without) work correctly

## Notes

These sparse data scenarios reflect real-world use cases:

1. **Legacy data**: Older recordings before event detection was implemented
2. **Failed detection**: Event detection algorithms failed due to data quality
3. **Selective analysis**: User chose not to run event detection on all data
4. **Mixed experiments**: Some conditions have events, others only traces

The tool must handle all these scenarios robustly to be production-ready.
