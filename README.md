# ideas-neural-circuit-analysis

Toolkit containing Inscopix secondary analysis tools available on [IDEAS](https://inscopix.github.io/ideas-docs/tools/index.html).

## Neural Circuit Correlations

### Single Study Analysis Tool

To run the [correlations tool](https://inscopix.github.io/ideas-docs/tools/epochs/correlations__correlation_tool/correlations__correlation_tool.html), execute the following CLI command:
```
ideas tools run correlations --inputs data/correlations/inputs.json 
```

### Combine and Compare Analysis Tool

To run the [combine and compare correlations tool](https://inscopix.github.io/ideas-docs/tools/standard-python/combine_compare_correlation_data/combine_compare_correlation_data.html), execute the following CLI command:
```
ideas tools run combine_compare_correlation_data --inputs data/combine_compare_correlation_data/inputs.json 
```

## Neural Population Activity

### Single Study Analysis Tool

To run the [population activity tool](https://inscopix.github.io/ideas-docs/tools/epochs/population_activity__population_activity/population_activity__population_activity.html), execute the following CLI command:
```
ideas tools run population_activity --inputs data/population_activity/inputs.json 
```

### Combine and Compare Analysis Tool

To run the [combine and compare population activity tool](https://inscopix.github.io/ideas-docs/tools/standard-python/combine_compare_population_data/combine_compare_population_data.html), execute the following CLI command:
```
ideas tools run combine_compare_population_activity_data --inputs data/combine_compare_population_activity_data/inputs.json 
```

## Peri-Event

### Single Study Analysis Tool

To run the [peri-event tool](https://inscopix.github.io/ideas-docs/tools/standard-python/peri_event_workflow/peri_event_workflow.html), execute the following CLI command:
```
ideas tools run peri_event_workflow --inputs data/peri_event_workflow/inputs.json
```

### Combine and Compare Analysis Tool

To run the [combine and compare peri-event data tool](https://inscopix.github.io/ideas-docs/tools/standard-python/combine_compare_peri_event_data/combine_compare_peri_event_data.html), execute the following CLI command:
```
ideas tools run combine_compare_peri_event_data --inputs data/combine_compare_peri_event_data/inputs.json 
```

## Peri-Event Across Epochs

## Single Study Analysis Tool

To run the [compare peri-event activity across epochs tool](https://inscopix.github.io/ideas-docs/tools/standard-python/compare_peri_event_activity_across_epochs/compare_peri_event_activity_across_epochs.html), execute the following CLI command:
```
ideas tools run compare_peri_event_activity_across_epochs -s
```

## Combine and Compare Analysis Tool

To run the [combine and compare peri-event data across epochs tool](https://inscopix.github.io/ideas-docs/tools/standard-python/combine_compare_peri_event_data_across_epochs/combine_compare_peri_event_data_across_epochs.html), execute the following CLI command:
```
ideas tools run compare_peri_event_activity_across_epochs -s
```

## Epoch Activity

## Single Study Analysis Tool

To run the [compare neural activity across epochs tool](https://inscopix.github.io/ideas-docs/tools/epochs/epoch_activity__run/epoch_activity__run.html), execute the following CLI command:
```
ideas tools run epoch_activity__run -s
```

## Combine and Compare Analysis Tool

To run the [combine and compare neural activity across epochs tool](https://inscopix.github.io/ideas-docs/tools/epochs/comb_comp_epochs__run_cc_epochs/comb_comp_epochs__run_cc_epochs.html), execute the following CLI command:
```
ideas tools run epoch_activity__run -s
```

## State Epoch Activity

## Single Study Analysis Tool

To run the [compare neural state data across epochs tool](https://inscopix.github.io/ideas-docs/tools/epochs/state_epoch_baseline/state_epoch_baseline.html), execute the following CLI command:
```
ideas tools run state_epoch_baseline -s
```

## Combine and Compare Analysis Tool

To run the [compare neural state-epoch data across between groups tool](https://inscopix.github.io/ideas-docs/tools/epochs/combine_compare_state_epoch_data/combine_compare_state_epoch_data.html), execute the following CLI command:
```
ideas tools run combine_compare_state_epoch_data -s
```
