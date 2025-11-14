import os
import shutil
import pandas as pd
import numpy as np
import unittest
from analysis.compare_peri_event_activity_across_epochs import (
    compare_peri_event_activity_across_epochs,
)
from ideas.exceptions import IdeasError

import pytest


@pytest.mark.skip(reason="This test case is temporarily disabled.")
class TestComparePeriEventActivityAcrossEpochs(unittest.TestCase):
    """Tests for the compare peri-event activity across epochs tool."""

    # tool info
    tool_key = "compare_peri_event_activity_across_epochs"

    # define directories
    temporary_dir = "/tmp"
    input_dir = "data/peri_event_workflow"
    output_dir = os.path.join(
        temporary_dir, "tmp_compare_peri_event_activity_across_epochs_outputs"
    )

    # output manifest
    output_manifest_json_schema = (
        "toolbox/tests/schemas/output_manifest_schema.json"
    )
    output_manifest_file_basename = "output_manifest.json"
    output_manifest_file = os.path.join(
        output_dir, output_manifest_file_basename
    )

    # output metadata
    output_metadata_json_schema = (
        "toolbox/tests/schemas/output_metadata_schema.json"
    )
    output_metadata_file_basename = "output_metadata.json"
    output_metadata_file = os.path.join(
        output_dir, output_metadata_file_basename
    )

    # input files
    input_cellset_isxd_files = [
        os.path.join(input_dir, "cellset_series_part1-PCA-ICA.isxd")
    ]
    input_events_h5_file = os.path.join(input_dir, "events.h5")

    def setUp(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        try:
            os.remove("exit_status.txt")
        except:
            pass

    def validate_existence_of_output_files(
        self, output_dir, files_to_exclude=[]
    ):
        """Validate that the expected output files exist once tool execution completes."""
        output_files = os.listdir(output_dir)
        for f in [
            # event-aligned traces
            "event_aligned_activity.TRACES.csv",
            "population_activity.preview.svg",
            "event_aligned_population_activity.preview.svg",
            "event_aligned_population_activity_Baseline.preview.svg",
            "event_aligned_population_activity_EarlyDrug.preview.svg",
            "event_aligned_population_activity_LateDrug.preview.svg",
            "event_aligned_single_cell_activity_heatmap_Baseline.preview.svg",
            "event_aligned_single_cell_activity_heatmap_EarlyDrug.preview.svg",
            "event_aligned_single_cell_activity_heatmap_LateDrug.preview.svg",
            # event-aligned statistics
            "event_aligned_activity.STATISTICS.csv",
            "event_aligned_activity_by_modulation_Baseline.preview.svg",
            "event_aligned_activity_by_modulation_EarlyDrug.preview.svg",
            "event_aligned_activity_by_modulation_LateDrug.preview.svg",
            "cell_map_Baseline.preview.svg",
            "cell_map_EarlyDrug.preview.svg",
            "cell_map_LateDrug.preview.svg",
            "event_aligned_activity_up_modulated.svg",
            "event_aligned_activity_down_modulated.svg",
            "event_aligned_activity_non_modulated.svg",
            "event_count_per_epoch.preview.svg",
            "num_modulated_cells_per_epoch.svg",
            "mean_post_minus_pre_activity_per_epoch.svg",
            # event-aligned epoch comparison data
            "pairwise_epoch_comparisons.csv",
            "post_minus_pre_differences_Baseline_EarlyDrug.svg",
            "post_minus_pre_differences_Baseline_LateDrug.svg",
            "post_minus_pre_differences_EarlyDrug_LateDrug.svg",
            "post_minus_pre_boxplot.svg",
            # output manifest & metadata
            # "output_manifest.json",
            # "output_metadata.json",
        ]:
            if f not in files_to_exclude:
                self.assertTrue(f in output_files, f"Output file {f} does not exist")

    def validate_traces_file_column_names(
        self, traces_df, epoch_names, cell_ids
    ):
        """Validate output traces file column names."""
        exp_epoch_headers = epoch_names + ["Time"]
        act_epochs_headers = np.unique(
            traces_df.columns.get_level_values(0)
        ).tolist()
        self.assertEqual(exp_epoch_headers, act_epochs_headers)
        self.assertEqual(traces_df.columns[0], ("Time", "Unnamed: 0_level_1"))

        for second_level_header in (
            [
                "population_mean",
                "population_sem",
                "population_shuffled_mean",
                "population_shuffled_lower_conf",
                "population_shuffled_upper_conf",
                "up_modulated_mean",
                "up_modulated_sem",
                "down_modulated_mean",
                "down_modulated_sem",
                "non_modulated_mean",
                "non_modulated_sem",
            ]
            + [f"C{str(i).zfill(2)}_mean" for i in cell_ids]
            + [f"C{str(i).zfill(2)}_sem" for i in cell_ids]
        ):
            for epoch_name in epoch_names:
                self.assertTrue(
                    (epoch_name, second_level_header) in traces_df.columns
                )

    def validate_statistics_file_column_names(self, df):
        """Validate output statistics file column names."""
        exp_cols = [
            "name",
            "epoch",
            "true_mean_post-pre",
            "shuffled_mean_post-pre",
            "shuffled_std_post-pre",
            "z-score",
            "p-value",
            "modulation",
        ]
        act_cols = list(df.columns)
        self.assertEqual(exp_cols, act_cols)

    def validate_pairwise_comparisons_file_column_names(
        self, df, cols_to_exclude=None
    ):
        """Validate output pairwise comparisons file column names."""
        exp_cols = [
            "Comparison",
            "Contrast",
            "A",
            "B",
            "Paired",
            "Parametric",
            "T",
            "dof",
            "alternative",
            "p-unc",
            "BF10",
            "hedges",
        ]

        if cols_to_exclude is not None:
            for c in cols_to_exclude:
                exp_cols.remove(c)

        act_cols = list(df.columns)
        self.assertEqual(exp_cols, act_cols)

    def get_expected_output_manifest(self):
        """Returns the expected output manifest.
        * Note that this the expected output manifest returned by this function
          will not match all cases and should only be used in applicable test cases.
        """
        exp_manifest = {
            "schema_version": "2.0.0",
            "groups": [
                {
                    "group_key": "compare_peri_event_activity_across_epochs_output",
                    "group_type": "tool_output",
                    "group_id": "94fbbf32-25ff-4509-aec7-2b2edca122b9",
                    "series": [],
                    "files": [
                        {
                            "file_key": "input_cellset_files",
                            "file_name": "cellset_series_part1-PCA-ICA.isxd",
                            "file_id": "ed1b62b0-883b-4738-85e6-e64f6e177418",
                            "file_path": "/ideas/toolbox/tests/data/peri_event_workflow/cellset_series_part1-PCA-ICA.isxd",
                            "file_type": "cell_set",
                            "file_format": "isxd",
                            "file_structure": "binary",
                            "file_category": "source",
                        },
                        {
                            "file_key": "input_events_h5_file",
                            "file_name": "events.h5",
                            "file_id": "c8e6fb09-7267-4643-937c-8eb15f9c4789",
                            "file_path": "/ideas/toolbox/tests/data/peri_event_workflow/events.h5",
                            "file_type": "timestamp_events",
                            "file_format": "h5",
                            "file_structure": "sparse_time_series",
                            "file_category": "source",
                        },
                        {
                            "file_key": "event_aligned_traces",
                            "file_name": "event_aligned_activity.TRACES.csv",
                            "file_id": "397a2acc-dd9c-4a4b-98dd-a5d4c62753d2",
                            "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity.TRACES.csv",
                            "file_type": "event_aligned_neural_data",
                            "file_format": "csv",
                            "file_structure": "time_series",
                            "file_category": "result",
                            "parent_ids": [
                                "ed1b62b0-883b-4738-85e6-e64f6e177418"
                            ],
                            "preview": [
                                {
                                    "name": "Mean population activity",
                                    "help": "Mean population activity over time. Shaded areas represent the different epochs. Individual events are displayed underneath the population activity trace.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/population_activity.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity across epochs",
                                    "help": "Comparison of event-aligned average population activity across the epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity",
                                    "help": "Event-aligned average population activity line plot (epoch: Baseline).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity",
                                    "help": "Event-aligned average population activity line plot (epoch: Early Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity_EarlyDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity",
                                    "help": "Event-aligned average population activity line plot (epoch: Late Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity_LateDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned single-cell activity figure",
                                    "help": "Event-aligned single-cell activity heatmap (epoch: Baseline)",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_single_cell_activity_heatmap_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned single-cell activity figure",
                                    "help": "Event-aligned single-cell activity heatmap (epoch: Early Drug)",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_single_cell_activity_heatmap_EarlyDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned single-cell activity figure",
                                    "help": "Event-aligned single-cell activity heatmap (epoch: Late Drug)",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_single_cell_activity_heatmap_LateDrug.preview.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                        {
                            "file_key": "event_aligned_statistics",
                            "file_name": "event_aligned_activity.STATISTICS.csv",
                            "file_id": "4b96f018-98d0-40c9-981a-1db18e20b5a2",
                            "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity.STATISTICS.csv",
                            "file_type": "statistics",
                            "file_format": "csv",
                            "file_structure": "table",
                            "file_category": "result",
                            "parent_ids": [
                                "ed1b62b0-883b-4738-85e6-e64f6e177418",
                                "397a2acc-dd9c-4a4b-98dd-a5d4c62753d2",
                            ],
                            "preview": [
                                {
                                    "name": "Event-aligned sub-population activity figure",
                                    "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Baseline).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_by_modulation_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned sub-population activity figure",
                                    "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Early Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_by_modulation_EarlyDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned sub-population activity figure",
                                    "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Late Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_by_modulation_LateDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Spatial organization of modulation",
                                    "help": "Cell map visualizing spatial organization of modulation (epoch: Baseline).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/cell_map_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Spatial organization of modulation",
                                    "help": "Cell map visualizing spatial organization of modulation (epoch: Early Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/cell_map_EarlyDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Spatial organization of modulation",
                                    "help": "Cell map visualizing spatial organization of modulation (epoch: Late Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/cell_map_LateDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned activity of up-modulated cells",
                                    "help": "Comparison of event-aligned activity of up-modulated cells across epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_up_modulated.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned activity of down-modulated cells",
                                    "help": "Comparison of event-aligned activity of down-modulated cells across epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_down_modulated.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned activity of non-modulated cells",
                                    "help": "Comparison of event-aligned activity of non-modulated cells across epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_non_modulated.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Number of modulated cells per epoch",
                                    "help": "Number of up-, down-, and non-modulated neurons per epoch.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/num_modulated_cells_per_epoch.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Number of events per epoch",
                                    "help": "Number of events in each epoch.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_count_per_epoch.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Mean post-pre activity per epoch",
                                    "help": "Comparison of mean post-pre activity across the epochs. The error bars represent the standard error of the mean.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/mean_post_minus_pre_activity_per_epoch.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                        {
                            "file_key": "event_aligned_epoch_comparison_data",
                            "file_name": "pairwise_epoch_comparisons.csv",
                            "file_id": "bfc9d1f3-7f38-4578-9549-d2eb4a1eff2a",
                            "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/pairwise_epoch_comparisons.csv",
                            "file_type": "peri_event_comparison_data",
                            "file_format": "csv",
                            "file_structure": "table",
                            "file_category": "result",
                            "parent_ids": [
                                "ed1b62b0-883b-4738-85e6-e64f6e177418",
                                "397a2acc-dd9c-4a4b-98dd-a5d4c62753d2",
                                "4b96f018-98d0-40c9-981a-1db18e20b5a2",
                            ],
                            "preview": [
                                {
                                    "name": "Post-pre differences between Baseline and Early Drug",
                                    "help": "Pairwise difference of post-pre activity between epochs Baseline and Early Drug. The left panel presents the data as a histogram. The right panel contains a cell map colored by the magnitude of the difference in post-pre activity between the epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/post_minus_pre_differences_Baseline_EarlyDrug.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Post-pre differences between Baseline and Late Drug",
                                    "help": "Pairwise difference of post-pre activity between epochs Baseline and Late Drug. The left panel presents the data as a histogram. The right panel contains a cell map colored by the magnitude of the difference in post-pre activity between the epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/post_minus_pre_differences_Baseline_LateDrug.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Post-pre differences between Early Drug and Late Drug",
                                    "help": "Pairwise difference of post-pre activity between epochs Early Drug and Late Drug. The left panel presents the data as a histogram. The right panel contains a cell map colored by the magnitude of the difference in post-pre activity between the epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/post_minus_pre_differences_EarlyDrug_LateDrug.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Post-pre distribution across the epochs",
                                    "help": "Distribution of post-pre activity across epochs displayed using a box plot. Lines connect the same cells together.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/post_minus_pre_boxplot.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                    ],
                }
            ],
        }
        return exp_manifest

    def test_compare_peri_event_activity_across_epochs_single_cell_set(
        self,
    ):
        """
        Validate that the compare peri-event activity across epoch tool
        completes successfully with 3 epochs, each containing at least
        one valid event.
        """

        # define input parameters
        epoch_names = ["Baseline", "Early Drug", "Late Drug"]
        input_parameters = {
            "event_types": ["event_type_1"],
            "visual_window": {"pre": -2, "post": 2},
            "statistical_window": {"pre": [-1, 0], "post": [0, 1]},
            "num_shuffles": 50,
            "significance_threshold": 0.05,
            "seed": 0,
            "define_epochs_by": "global file time",
            "epoch_names": ", ".join(epoch_names),
            "epoch_periods": "(0, 61), (70, 105), (110, 130)",
            "epoch_colors": "tab:grey, tab:blue, tab:cyan",
            "epoch_min_events": 1,
        }

        # execute workflow
        compare_peri_event_activity_across_epochs(
            input_cellset_files=self.input_cellset_isxd_files,
            input_events_h5_file=self.input_events_h5_file,
            event_type=input_parameters["event_types"][0],
            visual_window_pre=input_parameters["visual_window"]["pre"],
            visual_window_post=input_parameters["visual_window"]["post"],
            statistical_window_pre_start=input_parameters[
                "statistical_window"
            ]["pre"][0],
            statistical_window_pre_end=input_parameters["statistical_window"][
                "pre"
            ][1],
            statistical_window_post_start=input_parameters[
                "statistical_window"
            ]["post"][0],
            statistical_window_post_end=input_parameters["statistical_window"][
                "post"
            ][1],
            num_shuffles=input_parameters["num_shuffles"],
            significance_threshold=input_parameters["significance_threshold"],
            seed=input_parameters["seed"],
            define_epochs_by=input_parameters["define_epochs_by"],
            epoch_names=input_parameters["epoch_names"],
            epoch_periods=input_parameters["epoch_periods"],
            epoch_colors=input_parameters["epoch_colors"],
            epoch_min_events=input_parameters["epoch_min_events"],
            output_dir=self.output_dir,
        )

        # validate existence of output files
        self.validate_existence_of_output_files(output_dir=self.output_dir)

        # validate peri-event traces csv file
        traces_csv_file = os.path.join(
            self.output_dir, "event_aligned_activity.TRACES.csv"
        )
        traces_df = pd.read_csv(traces_csv_file, header=[0, 1])
        self.assertEqual(traces_df.shape, (41, 154))
        self.validate_traces_file_column_names(
            traces_df=traces_df, epoch_names=epoch_names, cell_ids=range(20)
        )

        # validate peri-event statistics file
        statistics_csv_file = os.path.join(
            self.output_dir, "event_aligned_activity.STATISTICS.csv"
        )
        statistics_df = pd.read_csv(statistics_csv_file)
        self.assertEqual(statistics_df.shape, (72, 8))
        self.validate_statistics_file_column_names(statistics_df)

        # validate epoch comparison data file
        comparison_csv_file = os.path.join(
            self.output_dir, "pairwise_epoch_comparisons.csv"
        )
        comparison_df = pd.read_csv(comparison_csv_file)
        self.assertEqual(comparison_df.shape, (3, 12))
        self.validate_pairwise_comparisons_file_column_names(comparison_df)

        # # validate output manifest
        # self.assertTrue(
        #     validate_output_manifest(
        #         expected_output_manifest=self.get_expected_output_manifest(),
        #         tool_metadata=self.tool_metadata[self.tool_key],
        #         actual_manifest_data_filepath=self.output_manifest_file,
        #     )
        # )

    def test_compare_peri_event_activity_across_epochs_use_accepted_cells_only(
        self,
    ):
        """
        - Validate that the compare peri-event activity across epoch tool
        only uses accepted cells in the analysis.
        - The input cell set has the following 20 cells:
            - 9 accepted cells (should be the only ones used in the analysis)
            - 6 undecided cells (excluded from the analysis)
            - 5 rejected cells (excluded from the analysis)
        """

        # define input parameters
        epoch_names = ["Baseline", "Early Drug", "Late Drug"]
        input_parameters = {
            "event_types": ["event_type_1"],
            "visual_window": {"pre": -2, "post": 2},
            "statistical_window": {"pre": [-1, 0], "post": [0, 1]},
            "num_shuffles": 50,
            "significance_threshold": 0.05,
            "seed": 0,
            "define_epochs_by": "global file time",
            "epoch_names": ", ".join(epoch_names),
            "epoch_periods": "(0, 61), (70, 105), (110, 130)",
            "epoch_colors": "tab:grey, tab:blue, tab:cyan",
            "epoch_min_events": 1,
        }

        # execute workflow
        input_cellset_files = [
            os.path.join(
                self.input_dir,
                "cellset_accepted_undecided_and_rejected_cells.isxd",
            )
        ]
        compare_peri_event_activity_across_epochs(
            input_cellset_files=input_cellset_files,
            input_events_h5_file=self.input_events_h5_file,
            event_type=input_parameters["event_types"][0],
            visual_window_pre=input_parameters["visual_window"]["pre"],
            visual_window_post=input_parameters["visual_window"]["post"],
            statistical_window_pre_start=input_parameters[
                "statistical_window"
            ]["pre"][0],
            statistical_window_pre_end=input_parameters["statistical_window"][
                "pre"
            ][1],
            statistical_window_post_start=input_parameters[
                "statistical_window"
            ]["post"][0],
            statistical_window_post_end=input_parameters["statistical_window"][
                "post"
            ][1],
            num_shuffles=input_parameters["num_shuffles"],
            significance_threshold=input_parameters["significance_threshold"],
            seed=input_parameters["seed"],
            define_epochs_by=input_parameters["define_epochs_by"],
            epoch_names=input_parameters["epoch_names"],
            epoch_periods=input_parameters["epoch_periods"],
            epoch_colors=input_parameters["epoch_colors"],
            epoch_min_events=input_parameters["epoch_min_events"],
            output_dir=self.output_dir,
        )

        # validate existence of output files
        self.validate_existence_of_output_files(
            output_dir=self.output_dir,
            files_to_exclude=[
                # all cells are non-modulated in this case,
                # so no up- and down-modulated data to plot
                "event_aligned_activity_up_modulated.svg",
                "event_aligned_activity_down_modulated.svg",
            ],
        )

        # validate peri-event traces csv file
        traces_csv_file = os.path.join(
            self.output_dir, "event_aligned_activity.TRACES.csv"
        )
        traces_df = pd.read_csv(traces_csv_file, header=[0, 1])
        self.assertEqual(traces_df.shape, (41, 88))
        self.validate_traces_file_column_names(
            traces_df=traces_df,
            epoch_names=epoch_names,
            cell_ids=[0, 1, 3, 6, 7, 8, 13, 14, 18],
        )

        # validate peri-event statistics file
        statistics_csv_file = os.path.join(
            self.output_dir, "event_aligned_activity.STATISTICS.csv"
        )
        statistics_df = pd.read_csv(statistics_csv_file)
        self.assertEqual(statistics_df.shape, (39, 8))
        self.validate_statistics_file_column_names(statistics_df)

        # validate epoch comparison data file
        comparison_csv_file = os.path.join(
            self.output_dir, "pairwise_epoch_comparisons.csv"
        )
        comparison_df = pd.read_csv(comparison_csv_file)
        self.assertEqual(comparison_df.shape, (3, 12))
        self.validate_pairwise_comparisons_file_column_names(comparison_df)

        # validate output manifest
        exp_manifest = {
            "schema_version": "2.0.0",
            "groups": [
                {
                    "group_key": "compare_peri_event_activity_across_epochs_output",
                    "group_type": "tool_output",
                    "group_id": "94fbbf32-25ff-4509-aec7-2b2edca122b9",
                    "series": [],
                    "files": [
                        {
                            "file_key": "input_cellset_files",
                            "file_name": "cellset_accepted_undecided_and_rejected_cells.isxd",
                            "file_id": "ed1b62b0-883b-4738-85e6-e64f6e177418",
                            "file_path": "/ideas/toolbox/tests/data/peri_event_workflow/cellset_accepted_undecided_and_rejected_cells.isxd",
                            "file_type": "cell_set",
                            "file_format": "isxd",
                            "file_structure": "binary",
                            "file_category": "source",
                        },
                        {
                            "file_key": "input_events_h5_file",
                            "file_name": "events.h5",
                            "file_id": "c8e6fb09-7267-4643-937c-8eb15f9c4789",
                            "file_path": "/ideas/toolbox/tests/data/peri_event_workflow/events.h5",
                            "file_type": "timestamp_events",
                            "file_format": "h5",
                            "file_structure": "sparse_time_series",
                            "file_category": "source",
                        },
                        {
                            "file_key": "event_aligned_traces",
                            "file_name": "event_aligned_activity.TRACES.csv",
                            "file_id": "397a2acc-dd9c-4a4b-98dd-a5d4c62753d2",
                            "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity.TRACES.csv",
                            "file_type": "event_aligned_neural_data",
                            "file_format": "csv",
                            "file_structure": "time_series",
                            "file_category": "result",
                            "parent_ids": [
                                "ed1b62b0-883b-4738-85e6-e64f6e177418"
                            ],
                            "preview": [
                                {
                                    "name": "Mean population activity",
                                    "help": "Mean population activity over time. Shaded areas represent the different epochs. Individual events are displayed underneath the population activity trace.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/population_activity.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity across epochs",
                                    "help": "Comparison of event-aligned average population activity across the epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity",
                                    "help": "Event-aligned average population activity line plot (epoch: Baseline).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity",
                                    "help": "Event-aligned average population activity line plot (epoch: Early Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity_EarlyDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity",
                                    "help": "Event-aligned average population activity line plot (epoch: Late Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity_LateDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned single-cell activity figure",
                                    "help": "Event-aligned single-cell activity heatmap (epoch: Baseline)",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_single_cell_activity_heatmap_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned single-cell activity figure",
                                    "help": "Event-aligned single-cell activity heatmap (epoch: Early Drug)",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_single_cell_activity_heatmap_EarlyDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned single-cell activity figure",
                                    "help": "Event-aligned single-cell activity heatmap (epoch: Late Drug)",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_single_cell_activity_heatmap_LateDrug.preview.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                        {
                            "file_key": "event_aligned_statistics",
                            "file_name": "event_aligned_activity.STATISTICS.csv",
                            "file_id": "4b96f018-98d0-40c9-981a-1db18e20b5a2",
                            "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity.STATISTICS.csv",
                            "file_type": "statistics",
                            "file_format": "csv",
                            "file_structure": "table",
                            "file_category": "result",
                            "parent_ids": [
                                "ed1b62b0-883b-4738-85e6-e64f6e177418",
                                "397a2acc-dd9c-4a4b-98dd-a5d4c62753d2",
                            ],
                            "preview": [
                                {
                                    "name": "Event-aligned sub-population activity figure",
                                    "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Baseline).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_by_modulation_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned sub-population activity figure",
                                    "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Early Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_by_modulation_EarlyDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned sub-population activity figure",
                                    "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Late Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_by_modulation_LateDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Spatial organization of modulation",
                                    "help": "Cell map visualizing spatial organization of modulation (epoch: Baseline).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/cell_map_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Spatial organization of modulation",
                                    "help": "Cell map visualizing spatial organization of modulation (epoch: Early Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/cell_map_EarlyDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Spatial organization of modulation",
                                    "help": "Cell map visualizing spatial organization of modulation (epoch: Late Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/cell_map_LateDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned activity of down-modulated cells",
                                    "help": "Comparison of event-aligned activity of down-modulated cells across epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_down_modulated.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned activity of non-modulated cells",
                                    "help": "Comparison of event-aligned activity of non-modulated cells across epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_non_modulated.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Number of modulated cells per epoch",
                                    "help": "Number of up-, down-, and non-modulated neurons per epoch.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/num_modulated_cells_per_epoch.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Number of events per epoch",
                                    "help": "Number of events in each epoch.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_count_per_epoch.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Mean post-pre activity per epoch",
                                    "help": "Comparison of mean post-pre activity across the epochs. The error bars represent the standard error of the mean.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/mean_post_minus_pre_activity_per_epoch.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                        {
                            "file_key": "event_aligned_epoch_comparison_data",
                            "file_name": "pairwise_epoch_comparisons.csv",
                            "file_id": "bfc9d1f3-7f38-4578-9549-d2eb4a1eff2a",
                            "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/pairwise_epoch_comparisons.csv",
                            "file_type": "peri_event_comparison_data",
                            "file_format": "csv",
                            "file_structure": "table",
                            "file_category": "result",
                            "parent_ids": [
                                "ed1b62b0-883b-4738-85e6-e64f6e177418",
                                "397a2acc-dd9c-4a4b-98dd-a5d4c62753d2",
                                "4b96f018-98d0-40c9-981a-1db18e20b5a2",
                            ],
                            "preview": [
                                {
                                    "name": "Post-pre differences between Baseline and Early Drug",
                                    "help": "Pairwise difference of post-pre activity between epochs Baseline and Early Drug. The left panel presents the data as a histogram. The right panel contains a cell map colored by the magnitude of the difference in post-pre activity between the epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/post_minus_pre_differences_Baseline_EarlyDrug.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Post-pre differences between Baseline and Late Drug",
                                    "help": "Pairwise difference of post-pre activity between epochs Baseline and Late Drug. The left panel presents the data as a histogram. The right panel contains a cell map colored by the magnitude of the difference in post-pre activity between the epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/post_minus_pre_differences_Baseline_LateDrug.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Post-pre differences between Early Drug and Late Drug",
                                    "help": "Pairwise difference of post-pre activity between epochs Early Drug and Late Drug. The left panel presents the data as a histogram. The right panel contains a cell map colored by the magnitude of the difference in post-pre activity between the epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/post_minus_pre_differences_EarlyDrug_LateDrug.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Post-pre distribution across the epochs",
                                    "help": "Distribution of post-pre activity across epochs displayed using a box plot. Lines connect the same cells together.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/post_minus_pre_boxplot.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                    ],
                }
            ],
        }
        # # validate output manifest
        # self.assertTrue(
        #     validate_output_manifest(
        #         expected_output_manifest=exp_manifest,
        #         tool_metadata=self.tool_metadata[self.tool_key],
        #         actual_manifest_data_filepath=self.output_manifest_file,
        #     )
        # )

    def test_compare_peri_event_activity_across_epochs_use_undecided_cells_only(
        self,
    ):
        """
        - Validate that the compare peri-event activity across epoch tool
        only uses undecided when no accepted cells are detected
        - The input cell set has 20 undecided cells.
        """

        # define input parameters
        epoch_names = ["Baseline", "Early Drug", "Late Drug"]
        input_parameters = {
            "event_types": ["event_type_1"],
            "visual_window": {"pre": -2, "post": 2},
            "statistical_window": {"pre": [-1, 0], "post": [0, 1]},
            "num_shuffles": 50,
            "significance_threshold": 0.05,
            "seed": 0,
            "define_epochs_by": "global file time",
            "epoch_names": ", ".join(epoch_names),
            "epoch_periods": "(0, 61), (70, 105), (110, 130)",
            "epoch_colors": "tab:grey, tab:blue, tab:cyan",
            "epoch_min_events": 1,
        }

        # execute workflow
        input_cellset_files = [
            os.path.join(self.input_dir, "cellset_undecided_cells_only.isxd")
        ]
        compare_peri_event_activity_across_epochs(
            input_cellset_files=input_cellset_files,
            input_events_h5_file=self.input_events_h5_file,
            event_type=input_parameters["event_types"][0],
            visual_window_pre=input_parameters["visual_window"]["pre"],
            visual_window_post=input_parameters["visual_window"]["post"],
            statistical_window_pre_start=input_parameters[
                "statistical_window"
            ]["pre"][0],
            statistical_window_pre_end=input_parameters["statistical_window"][
                "pre"
            ][1],
            statistical_window_post_start=input_parameters[
                "statistical_window"
            ]["post"][0],
            statistical_window_post_end=input_parameters["statistical_window"][
                "post"
            ][1],
            num_shuffles=input_parameters["num_shuffles"],
            significance_threshold=input_parameters["significance_threshold"],
            seed=input_parameters["seed"],
            define_epochs_by=input_parameters["define_epochs_by"],
            epoch_names=input_parameters["epoch_names"],
            epoch_periods=input_parameters["epoch_periods"],
            epoch_colors=input_parameters["epoch_colors"],
            epoch_min_events=input_parameters["epoch_min_events"],
            output_dir=self.output_dir,
        )

        # validate existence of output files
        self.validate_existence_of_output_files(output_dir=self.output_dir)

        # validate peri-event traces csv file
        traces_csv_file = os.path.join(
            self.output_dir, "event_aligned_activity.TRACES.csv"
        )
        traces_df = pd.read_csv(traces_csv_file, header=[0, 1])
        self.assertEqual(traces_df.shape, (41, 154))
        self.validate_traces_file_column_names(
            traces_df=traces_df, epoch_names=epoch_names, cell_ids=range(20)
        )

        # validate peri-event statistics file
        statistics_csv_file = os.path.join(
            self.output_dir, "event_aligned_activity.STATISTICS.csv"
        )
        statistics_df = pd.read_csv(statistics_csv_file)
        self.assertEqual(statistics_df.shape, (72, 8))
        self.validate_statistics_file_column_names(statistics_df)

        # validate epoch comparison data file
        comparison_csv_file = os.path.join(
            self.output_dir, "pairwise_epoch_comparisons.csv"
        )
        comparison_df = pd.read_csv(comparison_csv_file)
        self.assertEqual(comparison_df.shape, (3, 12))
        self.validate_pairwise_comparisons_file_column_names(comparison_df)

        # validate output manifest
        exp_manifest = {
            "schema_version": "2.0.0",
            "groups": [
                {
                    "group_key": "compare_peri_event_activity_across_epochs_output",
                    "group_type": "tool_output",
                    "group_id": "94fbbf32-25ff-4509-aec7-2b2edca122b9",
                    "series": [],
                    "files": [
                        {
                            "file_key": "input_cellset_files",
                            "file_name": "cellset_undecided_cells_only.isxd",
                            "file_id": "ed1b62b0-883b-4738-85e6-e64f6e177418",
                            "file_path": "/ideas/toolbox/tests/data/peri_event_workflow/cellset_undecided_cells_only.isxd",
                            "file_type": "cell_set",
                            "file_format": "isxd",
                            "file_structure": "binary",
                            "file_category": "source",
                        },
                        {
                            "file_key": "input_events_h5_file",
                            "file_name": "events.h5",
                            "file_id": "c8e6fb09-7267-4643-937c-8eb15f9c4789",
                            "file_path": "/ideas/toolbox/tests/data/peri_event_workflow/events.h5",
                            "file_type": "timestamp_events",
                            "file_format": "h5",
                            "file_structure": "sparse_time_series",
                            "file_category": "source",
                        },
                        {
                            "file_key": "event_aligned_traces",
                            "file_name": "event_aligned_activity.TRACES.csv",
                            "file_id": "397a2acc-dd9c-4a4b-98dd-a5d4c62753d2",
                            "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity.TRACES.csv",
                            "file_type": "event_aligned_neural_data",
                            "file_format": "csv",
                            "file_structure": "time_series",
                            "file_category": "result",
                            "parent_ids": [
                                "ed1b62b0-883b-4738-85e6-e64f6e177418"
                            ],
                            "preview": [
                                {
                                    "name": "Mean population activity",
                                    "help": "Mean population activity over time. Shaded areas represent the different epochs. Individual events are displayed underneath the population activity trace.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/population_activity.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity across epochs",
                                    "help": "Comparison of event-aligned average population activity across the epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity",
                                    "help": "Event-aligned average population activity line plot (epoch: Baseline).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity",
                                    "help": "Event-aligned average population activity line plot (epoch: Early Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity_EarlyDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity",
                                    "help": "Event-aligned average population activity line plot (epoch: Late Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity_LateDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned single-cell activity figure",
                                    "help": "Event-aligned single-cell activity heatmap (epoch: Baseline)",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_single_cell_activity_heatmap_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned single-cell activity figure",
                                    "help": "Event-aligned single-cell activity heatmap (epoch: Early Drug)",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_single_cell_activity_heatmap_EarlyDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned single-cell activity figure",
                                    "help": "Event-aligned single-cell activity heatmap (epoch: Late Drug)",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_single_cell_activity_heatmap_LateDrug.preview.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                        {
                            "file_key": "event_aligned_statistics",
                            "file_name": "event_aligned_activity.STATISTICS.csv",
                            "file_id": "4b96f018-98d0-40c9-981a-1db18e20b5a2",
                            "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity.STATISTICS.csv",
                            "file_type": "statistics",
                            "file_format": "csv",
                            "file_structure": "table",
                            "file_category": "result",
                            "parent_ids": [
                                "ed1b62b0-883b-4738-85e6-e64f6e177418",
                                "397a2acc-dd9c-4a4b-98dd-a5d4c62753d2",
                            ],
                            "preview": [
                                {
                                    "name": "Event-aligned sub-population activity figure",
                                    "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Baseline).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_by_modulation_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned sub-population activity figure",
                                    "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Early Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_by_modulation_EarlyDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned sub-population activity figure",
                                    "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Late Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_by_modulation_LateDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Spatial organization of modulation",
                                    "help": "Cell map visualizing spatial organization of modulation (epoch: Baseline).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/cell_map_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Spatial organization of modulation",
                                    "help": "Cell map visualizing spatial organization of modulation (epoch: Early Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/cell_map_EarlyDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Spatial organization of modulation",
                                    "help": "Cell map visualizing spatial organization of modulation (epoch: Late Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/cell_map_LateDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned activity of up-modulated cells",
                                    "help": "Comparison of event-aligned activity of up-modulated cells across epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_up_modulated.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned activity of down-modulated cells",
                                    "help": "Comparison of event-aligned activity of down-modulated cells across epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_down_modulated.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned activity of non-modulated cells",
                                    "help": "Comparison of event-aligned activity of non-modulated cells across epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_non_modulated.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Number of modulated cells per epoch",
                                    "help": "Number of up-, down-, and non-modulated neurons per epoch.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/num_modulated_cells_per_epoch.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Number of events per epoch",
                                    "help": "Number of events in each epoch.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_count_per_epoch.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Mean post-pre activity per epoch",
                                    "help": "Comparison of mean post-pre activity across the epochs. The error bars represent the standard error of the mean.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/mean_post_minus_pre_activity_per_epoch.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                        {
                            "file_key": "event_aligned_epoch_comparison_data",
                            "file_name": "pairwise_epoch_comparisons.csv",
                            "file_id": "bfc9d1f3-7f38-4578-9549-d2eb4a1eff2a",
                            "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/pairwise_epoch_comparisons.csv",
                            "file_type": "peri_event_comparison_data",
                            "file_format": "csv",
                            "file_structure": "table",
                            "file_category": "result",
                            "parent_ids": [
                                "ed1b62b0-883b-4738-85e6-e64f6e177418",
                                "397a2acc-dd9c-4a4b-98dd-a5d4c62753d2",
                                "4b96f018-98d0-40c9-981a-1db18e20b5a2",
                            ],
                            "preview": [
                                {
                                    "name": "Post-pre differences between Baseline and Early Drug",
                                    "help": "Pairwise difference of post-pre activity between epochs Baseline and Early Drug. The left panel presents the data as a histogram. The right panel contains a cell map colored by the magnitude of the difference in post-pre activity between the epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/post_minus_pre_differences_Baseline_EarlyDrug.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Post-pre differences between Baseline and Late Drug",
                                    "help": "Pairwise difference of post-pre activity between epochs Baseline and Late Drug. The left panel presents the data as a histogram. The right panel contains a cell map colored by the magnitude of the difference in post-pre activity between the epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/post_minus_pre_differences_Baseline_LateDrug.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Post-pre differences between Early Drug and Late Drug",
                                    "help": "Pairwise difference of post-pre activity between epochs Early Drug and Late Drug. The left panel presents the data as a histogram. The right panel contains a cell map colored by the magnitude of the difference in post-pre activity between the epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/post_minus_pre_differences_EarlyDrug_LateDrug.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Post-pre distribution across the epochs",
                                    "help": "Distribution of post-pre activity across epochs displayed using a box plot. Lines connect the same cells together.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/post_minus_pre_boxplot.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                    ],
                }
            ],
        }
        # self.assertTrue(
        #     validate_output_manifest(
        #         expected_output_manifest=exp_manifest,
        #         tool_metadata=self.tool_metadata[self.tool_key],
        #         actual_manifest_data_filepath=self.output_manifest_file,
        #     )
        # )

    def test_compare_peri_event_activity_across_epochs_single_accepted_cell(
        self,
    ):
        """
        Validate that the compare peri-event activity across epoch tool
        can handle the scenario where the input cell set contains a single accepted cell.
        """

        # define input parameters
        epoch_names = ["Baseline", "Early Drug", "Late Drug"]
        input_parameters = {
            "event_types": ["event_type_1"],
            "visual_window": {"pre": -2, "post": 2},
            "statistical_window": {"pre": [-1, 0], "post": [0, 1]},
            "num_shuffles": 50,
            "significance_threshold": 0.05,
            "seed": 0,
            "define_epochs_by": "global file time",
            "epoch_names": ", ".join(epoch_names),
            "epoch_periods": "(0, 61), (70, 105), (110, 130)",
            "epoch_colors": "tab:grey, tab:blue, tab:cyan",
            "epoch_min_events": 1,
        }

        # execute workflow
        input_cellset_files = [
            os.path.join(self.input_dir, "cellset_single_accepted_cell.isxd")
        ]
        compare_peri_event_activity_across_epochs(
            input_cellset_files=input_cellset_files,
            input_events_h5_file=self.input_events_h5_file,
            event_type=input_parameters["event_types"][0],
            visual_window_pre=input_parameters["visual_window"]["pre"],
            visual_window_post=input_parameters["visual_window"]["post"],
            statistical_window_pre_start=input_parameters[
                "statistical_window"
            ]["pre"][0],
            statistical_window_pre_end=input_parameters["statistical_window"][
                "pre"
            ][1],
            statistical_window_post_start=input_parameters[
                "statistical_window"
            ]["post"][0],
            statistical_window_post_end=input_parameters["statistical_window"][
                "post"
            ][1],
            num_shuffles=input_parameters["num_shuffles"],
            significance_threshold=input_parameters["significance_threshold"],
            seed=input_parameters["seed"],
            define_epochs_by=input_parameters["define_epochs_by"],
            epoch_names=input_parameters["epoch_names"],
            epoch_periods=input_parameters["epoch_periods"],
            epoch_colors=input_parameters["epoch_colors"],
            epoch_min_events=input_parameters["epoch_min_events"],
            output_dir=self.output_dir,
        )

        # validate existence of output files
        self.validate_existence_of_output_files(
            output_dir=self.output_dir,
            files_to_exclude=[
                # all cells are non-modulated in this case,
                # so no up- and down-modulated data to plot
                "event_aligned_activity_up_modulated.svg",
                "event_aligned_activity_down_modulated.svg",
            ],
        )

        # validate peri-event traces csv file
        traces_csv_file = os.path.join(
            self.output_dir, "event_aligned_activity.TRACES.csv"
        )
        traces_df = pd.read_csv(traces_csv_file, header=[0, 1])
        self.assertEqual(traces_df.shape, (41, 40))
        self.validate_traces_file_column_names(
            traces_df=traces_df, epoch_names=epoch_names, cell_ids=[7]
        )

        # validate peri-event statistics file
        statistics_csv_file = os.path.join(
            self.output_dir, "event_aligned_activity.STATISTICS.csv"
        )
        statistics_df = pd.read_csv(statistics_csv_file)
        self.assertEqual(statistics_df.shape, (15, 8))
        self.validate_statistics_file_column_names(statistics_df)

        # validate epoch comparison data file
        comparison_csv_file = os.path.join(
            self.output_dir, "pairwise_epoch_comparisons.csv"
        )
        comparison_df = pd.read_csv(comparison_csv_file)
        self.assertEqual(comparison_df.shape, (3, 9))
        self.validate_pairwise_comparisons_file_column_names(
            df=comparison_df,
            cols_to_exclude=[
                "T",
                "p-unc",
                "hedges",
            ],  # single cell, so not enough data to compute stats
        )

        # validate output manifest
        exp_manifest = {
            "schema_version": "2.0.0",
            "groups": [
                {
                    "group_key": "compare_peri_event_activity_across_epochs_output",
                    "group_type": "tool_output",
                    "group_id": "94fbbf32-25ff-4509-aec7-2b2edca122b9",
                    "series": [],
                    "files": [
                        {
                            "file_key": "input_cellset_files",
                            "file_name": "cellset_single_accepted_cell.isxd",
                            "file_id": "ed1b62b0-883b-4738-85e6-e64f6e177418",
                            "file_path": "/ideas/toolbox/tests/data/peri_event_workflow/cellset_single_accepted_cell.isxd",
                            "file_type": "cell_set",
                            "file_format": "isxd",
                            "file_structure": "binary",
                            "file_category": "source",
                        },
                        {
                            "file_key": "input_events_h5_file",
                            "file_name": "events.h5",
                            "file_id": "c8e6fb09-7267-4643-937c-8eb15f9c4789",
                            "file_path": "/ideas/toolbox/tests/data/peri_event_workflow/events.h5",
                            "file_type": "timestamp_events",
                            "file_format": "h5",
                            "file_structure": "sparse_time_series",
                            "file_category": "source",
                        },
                        {
                            "file_key": "event_aligned_traces",
                            "file_name": "event_aligned_activity.TRACES.csv",
                            "file_id": "397a2acc-dd9c-4a4b-98dd-a5d4c62753d2",
                            "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity.TRACES.csv",
                            "file_type": "event_aligned_neural_data",
                            "file_format": "csv",
                            "file_structure": "time_series",
                            "file_category": "result",
                            "parent_ids": [
                                "ed1b62b0-883b-4738-85e6-e64f6e177418"
                            ],
                            "preview": [
                                {
                                    "name": "Mean population activity",
                                    "help": "Mean population activity over time. Shaded areas represent the different epochs. Individual events are displayed underneath the population activity trace.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/population_activity.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity across epochs",
                                    "help": "Comparison of event-aligned average population activity across the epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity",
                                    "help": "Event-aligned average population activity line plot (epoch: Baseline).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity",
                                    "help": "Event-aligned average population activity line plot (epoch: Early Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity_EarlyDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity",
                                    "help": "Event-aligned average population activity line plot (epoch: Late Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity_LateDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned single-cell activity figure",
                                    "help": "Event-aligned single-cell activity heatmap (epoch: Baseline)",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_single_cell_activity_heatmap_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned single-cell activity figure",
                                    "help": "Event-aligned single-cell activity heatmap (epoch: Early Drug)",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_single_cell_activity_heatmap_EarlyDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned single-cell activity figure",
                                    "help": "Event-aligned single-cell activity heatmap (epoch: Late Drug)",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_single_cell_activity_heatmap_LateDrug.preview.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                        {
                            "file_key": "event_aligned_statistics",
                            "file_name": "event_aligned_activity.STATISTICS.csv",
                            "file_id": "4b96f018-98d0-40c9-981a-1db18e20b5a2",
                            "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity.STATISTICS.csv",
                            "file_type": "statistics",
                            "file_format": "csv",
                            "file_structure": "table",
                            "file_category": "result",
                            "parent_ids": [
                                "ed1b62b0-883b-4738-85e6-e64f6e177418",
                                "397a2acc-dd9c-4a4b-98dd-a5d4c62753d2",
                            ],
                            "preview": [
                                {
                                    "name": "Event-aligned sub-population activity figure",
                                    "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Baseline).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_by_modulation_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned sub-population activity figure",
                                    "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Early Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_by_modulation_EarlyDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned sub-population activity figure",
                                    "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Late Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_by_modulation_LateDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Spatial organization of modulation",
                                    "help": "Cell map visualizing spatial organization of modulation (epoch: Baseline).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/cell_map_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Spatial organization of modulation",
                                    "help": "Cell map visualizing spatial organization of modulation (epoch: Early Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/cell_map_EarlyDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Spatial organization of modulation",
                                    "help": "Cell map visualizing spatial organization of modulation (epoch: Late Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/cell_map_LateDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned activity of non-modulated cells",
                                    "help": "Comparison of event-aligned activity of non-modulated cells across epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_non_modulated.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Number of modulated cells per epoch",
                                    "help": "Number of up-, down-, and non-modulated neurons per epoch.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/num_modulated_cells_per_epoch.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Number of events per epoch",
                                    "help": "Number of events in each epoch.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_count_per_epoch.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Mean post-pre activity per epoch",
                                    "help": "Comparison of mean post-pre activity across the epochs. The error bars represent the standard error of the mean.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/mean_post_minus_pre_activity_per_epoch.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                        {
                            "file_key": "event_aligned_epoch_comparison_data",
                            "file_name": "pairwise_epoch_comparisons.csv",
                            "file_id": "bfc9d1f3-7f38-4578-9549-d2eb4a1eff2a",
                            "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/pairwise_epoch_comparisons.csv",
                            "file_type": "peri_event_comparison_data",
                            "file_format": "csv",
                            "file_structure": "table",
                            "file_category": "result",
                            "parent_ids": [
                                "ed1b62b0-883b-4738-85e6-e64f6e177418",
                                "397a2acc-dd9c-4a4b-98dd-a5d4c62753d2",
                                "4b96f018-98d0-40c9-981a-1db18e20b5a2",
                            ],
                            "preview": [
                                {
                                    "name": "Post-pre differences between Baseline and Early Drug",
                                    "help": "Pairwise difference of post-pre activity between epochs Baseline and Early Drug. The left panel presents the data as a histogram. The right panel contains a cell map colored by the magnitude of the difference in post-pre activity between the epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/post_minus_pre_differences_Baseline_EarlyDrug.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Post-pre differences between Baseline and Late Drug",
                                    "help": "Pairwise difference of post-pre activity between epochs Baseline and Late Drug. The left panel presents the data as a histogram. The right panel contains a cell map colored by the magnitude of the difference in post-pre activity between the epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/post_minus_pre_differences_Baseline_LateDrug.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Post-pre differences between Early Drug and Late Drug",
                                    "help": "Pairwise difference of post-pre activity between epochs Early Drug and Late Drug. The left panel presents the data as a histogram. The right panel contains a cell map colored by the magnitude of the difference in post-pre activity between the epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/post_minus_pre_differences_EarlyDrug_LateDrug.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Post-pre distribution across the epochs",
                                    "help": "Distribution of post-pre activity across epochs displayed using a box plot. Lines connect the same cells together.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/post_minus_pre_boxplot.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                    ],
                }
            ],
        }
        # validate output manifest
        # self.assertTrue(
        #     validate_output_manifest(
        #         expected_output_manifest=exp_manifest,
        #         tool_metadata=self.tool_metadata[self.tool_key],
        #         actual_manifest_data_filepath=self.output_manifest_file,
        #     )
        # )

    def test_compare_peri_event_activity_across_epochs_all_rejected_cells(
        self,
    ):
        """
        Validate that the compare peri-event activity across epoch tool
        raises an error when all cells in the input cell set are marked as rejected.
        """
        # define input parameters
        input_parameters = {
            "event_types": ["event_type_1"],
            "visual_window": {"pre": -2, "post": 2},
            "statistical_window": {"pre": [-1, 0], "post": [0, 1]},
            "num_shuffles": 50,
            "significance_threshold": 0.05,
            "seed": 0,
            "define_epochs_by": "global file time",
            "epoch_names": "Baseline, Early Drug, Late Drug",
            "epoch_periods": "(0, 2), (2.1, 5), (5.1, 9.9)",
            "epoch_colors": "tab:grey, tab:blue, tab:cyan",
            "epoch_min_events": 1,
        }

        input_cellset_files = [
            os.path.join(self.input_dir, "cellset_rejected_cells_only.isxd")
        ]

        self.assertRaisesRegex(
            IdeasError,
            "There are no accepted or undecided cells to process",
            compare_peri_event_activity_across_epochs,
            input_cellset_files=input_cellset_files,
            input_events_h5_file=self.input_events_h5_file,
            event_type=input_parameters["event_types"][0],
            visual_window_pre=input_parameters["visual_window"]["pre"],
            visual_window_post=input_parameters["visual_window"]["post"],
            statistical_window_pre_start=input_parameters[
                "statistical_window"
            ]["pre"][0],
            statistical_window_pre_end=input_parameters["statistical_window"][
                "pre"
            ][1],
            statistical_window_post_start=input_parameters[
                "statistical_window"
            ]["post"][0],
            statistical_window_post_end=input_parameters["statistical_window"][
                "post"
            ][1],
            num_shuffles=input_parameters["num_shuffles"],
            significance_threshold=input_parameters["significance_threshold"],
            seed=input_parameters["seed"],
            define_epochs_by=input_parameters["define_epochs_by"],
            epoch_names=input_parameters["epoch_names"],
            epoch_periods=input_parameters["epoch_periods"],
            epoch_colors=input_parameters["epoch_colors"],
            epoch_min_events=input_parameters["epoch_min_events"],
            output_dir=self.output_dir,
        )

    def test_compare_peri_event_activity_across_epochs_empty_epoch_excluded(
        self,
    ):
        """
        - Validate that the compare peri-event activity across epoch tool
        correctly excludes epochs that contain no events.
        - Here the 'Baseline' epoch contains no event and should be excluded from the analysis.
        """

        # define input parameters
        epoch_names = ["Baseline", "Early Drug", "Late Drug"]
        input_parameters = {
            "event_types": ["event_type_1"],
            "visual_window": {"pre": -2, "post": 2},
            "statistical_window": {"pre": [-1, 0], "post": [0, 1]},
            "num_shuffles": 50,
            "significance_threshold": 0.05,
            "seed": 0,
            "define_epochs_by": "global file time",
            "epoch_names": ", ".join(epoch_names),
            "epoch_periods": "(0, 30), (70, 105), (110, 130)",
            "epoch_colors": "tab:grey, tab:blue, tab:cyan",
            "epoch_min_events": 1,
        }

        # execute workflow
        compare_peri_event_activity_across_epochs(
            input_cellset_files=self.input_cellset_isxd_files,
            input_events_h5_file=self.input_events_h5_file,
            event_type=input_parameters["event_types"][0],
            visual_window_pre=input_parameters["visual_window"]["pre"],
            visual_window_post=input_parameters["visual_window"]["post"],
            statistical_window_pre_start=input_parameters[
                "statistical_window"
            ]["pre"][0],
            statistical_window_pre_end=input_parameters["statistical_window"][
                "pre"
            ][1],
            statistical_window_post_start=input_parameters[
                "statistical_window"
            ]["post"][0],
            statistical_window_post_end=input_parameters["statistical_window"][
                "post"
            ][1],
            num_shuffles=input_parameters["num_shuffles"],
            significance_threshold=input_parameters["significance_threshold"],
            seed=input_parameters["seed"],
            define_epochs_by=input_parameters["define_epochs_by"],
            epoch_names=input_parameters["epoch_names"],
            epoch_periods=input_parameters["epoch_periods"],
            epoch_colors=input_parameters["epoch_colors"],
            epoch_min_events=input_parameters["epoch_min_events"],
            output_dir=self.output_dir,
        )

        # validate existence of output files
        output_files = os.listdir(self.output_dir)
        self.validate_existence_of_output_files(
            output_dir=self.output_dir,
            files_to_exclude=[
                # all cells are non-modulated in this case,
                # so no up- and down-modulated data to plot
                "event_aligned_activity_up_modulated.svg",
                "event_aligned_activity_down_modulated.svg",
                # no Baseline epoch in this test case,
                # so we can exclude all output files expected for Baseline
                "event_aligned_population_activity_Baseline.preview.svg",
                "event_aligned_single_cell_activity_heatmap_Baseline.preview.svg",
                "event_aligned_activity_by_modulation_Baseline.preview.svg",
                "cell_map_Baseline.preview.svg",
                "post_minus_pre_differences_Baseline_EarlyDrug.svg",
                "post_minus_pre_differences_Baseline_LateDrug.svg",
                "post_minus_pre_differences_EarlyDrug_LateDrug.svg",
            ],
        )

        # ensure no files were produced for the Baseline epoch
        for f in [
            "event_aligned_population_activity_Baseline.preview.svg",
            "event_aligned_single_cell_activity_heatmap_Baseline.preview.svg",
            "event_aligned_activity_by_modulation_Baseline.preview.svg",
            "cell_map_Baseline.preview.svg",
        ]:
            self.assertTrue(f not in output_files)

        # validate peri-event traces csv file
        traces_csv_file = os.path.join(
            self.output_dir, "event_aligned_activity.TRACES.csv"
        )
        traces_df = pd.read_csv(traces_csv_file, header=[0, 1])
        self.assertEqual(traces_df.shape, (41, 103))
        self.validate_traces_file_column_names(
            traces_df=traces_df,
            epoch_names=["Early Drug", "Late Drug"],
            cell_ids=range(20),
        )

        # validate peri-event statistics file
        statistics_csv_file = os.path.join(
            self.output_dir, "event_aligned_activity.STATISTICS.csv"
        )
        statistics_df = pd.read_csv(statistics_csv_file)
        self.assertEqual(statistics_df.shape, (48, 8))
        self.validate_statistics_file_column_names(statistics_df)

        # validate epoch comparison data file
        comparison_csv_file = os.path.join(
            self.output_dir, "pairwise_epoch_comparisons.csv"
        )
        comparison_df = pd.read_csv(comparison_csv_file)
        self.assertEqual(comparison_df.shape, (1, 12))
        self.validate_pairwise_comparisons_file_column_names(comparison_df)

        # validate output manifest
        exp_manifest = {
            "schema_version": "2.0.0",
            "groups": [
                {
                    "group_key": "compare_peri_event_activity_across_epochs_output",
                    "group_type": "tool_output",
                    "group_id": "569a4746-ed86-46fe-83f4-c06de79544d7",
                    "series": [],
                    "files": [
                        {
                            "file_key": "input_cellset_files",
                            "file_name": "cellset_series_part1-PCA-ICA.isxd",
                            "file_id": "d119e5e1-c9e0-41cd-bca0-2333072c07ed",
                            "file_path": "/ideas/toolbox/tests/data/peri_event_workflow/cellset_series_part1-PCA-ICA.isxd",
                            "file_type": "cell_set",
                            "file_format": "isxd",
                            "file_structure": "binary",
                            "file_category": "source",
                        },
                        {
                            "file_key": "input_events_h5_file",
                            "file_name": "events.h5",
                            "file_id": "76dcff6f-997d-4f51-bdc5-551d1c0a64b2",
                            "file_path": "/ideas/toolbox/tests/data/peri_event_workflow/events.h5",
                            "file_type": "timestamp_events",
                            "file_format": "h5",
                            "file_structure": "sparse_time_series",
                            "file_category": "source",
                        },
                        {
                            "file_key": "event_aligned_traces",
                            "file_name": "event_aligned_activity.TRACES.csv",
                            "file_id": "e58bbe68-af93-421f-a990-74176db96656",
                            "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity.TRACES.csv",
                            "file_type": "event_aligned_neural_data",
                            "file_format": "csv",
                            "file_structure": "time_series",
                            "file_category": "result",
                            "parent_ids": [
                                "d119e5e1-c9e0-41cd-bca0-2333072c07ed"
                            ],
                            "preview": [
                                {
                                    "name": "Mean population activity",
                                    "help": "Mean population activity over time. Shaded areas represent the different epochs. Individual events are displayed underneath the population activity trace.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/population_activity.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity across epochs",
                                    "help": "Comparison of event-aligned average population activity across the epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity",
                                    "help": "Event-aligned average population activity line plot (epoch: Early Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity_EarlyDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity",
                                    "help": "Event-aligned average population activity line plot (epoch: Late Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity_LateDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned single-cell activity figure",
                                    "help": "Event-aligned single-cell activity heatmap (epoch: Early Drug)",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_single_cell_activity_heatmap_EarlyDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned single-cell activity figure",
                                    "help": "Event-aligned single-cell activity heatmap (epoch: Late Drug)",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_single_cell_activity_heatmap_LateDrug.preview.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                        {
                            "file_key": "event_aligned_statistics",
                            "file_name": "event_aligned_activity.STATISTICS.csv",
                            "file_id": "3ec42ba3-4178-456c-b728-cf2f2d59697f",
                            "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity.STATISTICS.csv",
                            "file_type": "statistics",
                            "file_format": "csv",
                            "file_structure": "table",
                            "file_category": "result",
                            "parent_ids": [
                                "d119e5e1-c9e0-41cd-bca0-2333072c07ed",
                                "e58bbe68-af93-421f-a990-74176db96656",
                            ],
                            "preview": [
                                {
                                    "name": "Event-aligned sub-population activity figure",
                                    "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Early Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_by_modulation_EarlyDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned sub-population activity figure",
                                    "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Late Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_by_modulation_LateDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Spatial organization of modulation",
                                    "help": "Cell map visualizing spatial organization of modulation (epoch: Early Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/cell_map_EarlyDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Spatial organization of modulation",
                                    "help": "Cell map visualizing spatial organization of modulation (epoch: Late Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/cell_map_LateDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned activity of down-modulated cells",
                                    "help": "Comparison of event-aligned activity of down-modulated cells across epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_down_modulated.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned activity of non-modulated cells",
                                    "help": "Comparison of event-aligned activity of non-modulated cells across epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_non_modulated.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Number of modulated cells per epoch",
                                    "help": "Number of up-, down-, and non-modulated neurons per epoch.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/num_modulated_cells_per_epoch.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Number of events per epoch",
                                    "help": "Number of events in each epoch.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_count_per_epoch.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Mean post-pre activity per epoch",
                                    "help": "Comparison of mean post-pre activity across the epochs. The error bars represent the standard error of the mean.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/mean_post_minus_pre_activity_per_epoch.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                        {
                            "file_key": "event_aligned_epoch_comparison_data",
                            "file_name": "pairwise_epoch_comparisons.csv",
                            "file_id": "cd72a660-368b-478a-ae83-b28bd55685d4",
                            "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/pairwise_epoch_comparisons.csv",
                            "file_type": "peri_event_comparison_data",
                            "file_format": "csv",
                            "file_structure": "table",
                            "file_category": "result",
                            "parent_ids": [
                                "d119e5e1-c9e0-41cd-bca0-2333072c07ed",
                                "e58bbe68-af93-421f-a990-74176db96656",
                                "3ec42ba3-4178-456c-b728-cf2f2d59697f",
                            ],
                            "preview": [
                                {
                                    "name": "Post-pre differences between Early Drug and Late Drug",
                                    "help": "Pairwise difference of post-pre activity between epochs Early Drug and Late Drug. The left panel presents the data as a histogram. The right panel contains a cell map colored by the magnitude of the difference in post-pre activity between the epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/post_minus_pre_differences_EarlyDrug_LateDrug.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Post-pre distribution across the epochs",
                                    "help": "Distribution of post-pre activity across epochs displayed using a box plot. Lines connect the same cells together.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/post_minus_pre_boxplot.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                    ],
                }
            ],
        }
        # self.assertTrue(
        #     validate_output_manifest(
        #         expected_output_manifest=exp_manifest,
        #         tool_metadata=self.tool_metadata[self.tool_key],
        #         actual_manifest_data_filepath=self.output_manifest_file,
        #     )
        # )

    def test_compare_peri_event_activity_across_epochs_no_events_found_in_any_epoch(
        self,
    ):
        """
        Validate that the compare peri-event activity across epoch tool
        raises an error when there are no epochs to analyze due to an
        insufficient number of event across all epochs.
        """

        # define input parameters
        input_parameters = {
            "event_types": ["event_type_1"],
            "visual_window": {"pre": -2, "post": 2},
            "statistical_window": {"pre": [-1, 0], "post": [0, 1]},
            "num_shuffles": 50,
            "significance_threshold": 0.05,
            "seed": 0,
            "define_epochs_by": "global file time",
            "epoch_names": "Baseline, Early Drug, Late Drug",
            "epoch_periods": "(0, 2), (2.1, 5), (5.1, 9.9)",
            "epoch_colors": "tab:grey, tab:blue, tab:cyan",
            "epoch_min_events": 1,
        }

        expected_error_message = (
            "There are no epochs to analyze. "
            "This may occur if all epochs were excluded from "
            "the analysis due to an insufficient number of events."
        )

        self.assertRaisesRegex(
            IdeasError,
            expected_error_message,
            compare_peri_event_activity_across_epochs,
            input_cellset_files=self.input_cellset_isxd_files,
            input_events_h5_file=self.input_events_h5_file,
            event_type=input_parameters["event_types"][0],
            visual_window_pre=input_parameters["visual_window"]["pre"],
            visual_window_post=input_parameters["visual_window"]["post"],
            statistical_window_pre_start=input_parameters[
                "statistical_window"
            ]["pre"][0],
            statistical_window_pre_end=input_parameters["statistical_window"][
                "pre"
            ][1],
            statistical_window_post_start=input_parameters[
                "statistical_window"
            ]["post"][0],
            statistical_window_post_end=input_parameters["statistical_window"][
                "post"
            ][1],
            num_shuffles=input_parameters["num_shuffles"],
            significance_threshold=input_parameters["significance_threshold"],
            seed=input_parameters["seed"],
            define_epochs_by=input_parameters["define_epochs_by"],
            epoch_names=input_parameters["epoch_names"],
            epoch_periods=input_parameters["epoch_periods"],
            epoch_colors=input_parameters["epoch_colors"],
            epoch_min_events=input_parameters["epoch_min_events"],
            output_dir=self.output_dir,
        )

    def test_compare_peri_event_activity_across_epochs_with_a_single_epoch(
        self,
    ):
        """
        Validate that the compare peri-event activity across epoch tool
        works as expected when analyzing data from a single epoch.
        """

        # define input parameters
        epoch_names = ["Baseline"]
        input_parameters = {
            "event_types": ["event_type_1"],
            "visual_window": {"pre": -2, "post": 2},
            "statistical_window": {"pre": [-1, 0], "post": [0, 1]},
            "num_shuffles": 50,
            "significance_threshold": 0.05,
            "seed": 0,
            "define_epochs_by": "global file time",
            "epoch_names": ", ".join(epoch_names),
            "epoch_periods": "(0, 61)",
            "epoch_colors": "tab:grey",
            "epoch_min_events": 1,
        }

        # execute workflow
        compare_peri_event_activity_across_epochs(
            input_cellset_files=self.input_cellset_isxd_files,
            input_events_h5_file=self.input_events_h5_file,
            event_type=input_parameters["event_types"][0],
            visual_window_pre=input_parameters["visual_window"]["pre"],
            visual_window_post=input_parameters["visual_window"]["post"],
            statistical_window_pre_start=input_parameters[
                "statistical_window"
            ]["pre"][0],
            statistical_window_pre_end=input_parameters["statistical_window"][
                "pre"
            ][1],
            statistical_window_post_start=input_parameters[
                "statistical_window"
            ]["post"][0],
            statistical_window_post_end=input_parameters["statistical_window"][
                "post"
            ][1],
            num_shuffles=input_parameters["num_shuffles"],
            significance_threshold=input_parameters["significance_threshold"],
            seed=input_parameters["seed"],
            define_epochs_by=input_parameters["define_epochs_by"],
            epoch_names=input_parameters["epoch_names"],
            epoch_periods=input_parameters["epoch_periods"],
            epoch_colors=input_parameters["epoch_colors"],
            epoch_min_events=input_parameters["epoch_min_events"],
            output_dir=self.output_dir,
        )

        # validate existence of output files
        output_files = os.listdir(self.output_dir)
        for f in [
            # event-aligned traces
            "event_aligned_activity.TRACES.csv",
            "population_activity.preview.svg",
            "event_aligned_population_activity.preview.svg",
            "event_aligned_population_activity_Baseline.preview.svg",
            "event_aligned_single_cell_activity_heatmap_Baseline.preview.svg",
            # event-aligned statistics
            "event_aligned_activity.STATISTICS.csv",
            "event_aligned_activity_by_modulation_Baseline.preview.svg",
            "cell_map_Baseline.preview.svg",
            "event_aligned_activity_non_modulated.svg",
            "event_count_per_epoch.preview.svg",
            "num_modulated_cells_per_epoch.svg",
            "mean_post_minus_pre_activity_per_epoch.svg",
            # event-aligned epoch comparison data --> no comparison data for 1 epoch
            # output manifest & metadata
            # "output_manifest.json",
            # "output_metadata.json",
        ]:
            self.assertTrue(f in output_files)

        # ensure comparison data does not exist
        assert "pairwise_epoch_comparisons.csv" not in output_files

        # validate peri-event traces csv file
        traces_csv_file = os.path.join(
            self.output_dir, "event_aligned_activity.TRACES.csv"
        )
        traces_df = pd.read_csv(traces_csv_file, header=[0, 1])
        self.assertEqual(traces_df.shape, (41, 52))
        self.validate_traces_file_column_names(
            traces_df=traces_df, epoch_names=epoch_names, cell_ids=range(20)
        )

        # validate peri-event statistics file
        statistics_csv_file = os.path.join(
            self.output_dir, "event_aligned_activity.STATISTICS.csv"
        )
        statistics_df = pd.read_csv(statistics_csv_file)
        self.assertEqual(statistics_df.shape, (24, 8))
        self.validate_statistics_file_column_names(statistics_df)

        # validate output manifest
        exp_manifest = {
            "schema_version": "2.0.0",
            "groups": [
                {
                    "group_key": "compare_peri_event_activity_across_epochs_output",
                    "group_type": "tool_output",
                    "group_id": "ae1e6eec-44cb-4091-b31d-0dab55725ab0",
                    "series": [],
                    "files": [
                        {
                            "file_key": "input_cellset_files",
                            "file_name": "cellset_series_part1-PCA-ICA.isxd",
                            "file_id": "d5821783-5f03-4184-a502-7c68ce080107",
                            "file_path": "/ideas/toolbox/tests/data/peri_event_workflow/cellset_series_part1-PCA-ICA.isxd",
                            "file_type": "cell_set",
                            "file_format": "isxd",
                            "file_structure": "binary",
                            "file_category": "source",
                        },
                        {
                            "file_key": "input_events_h5_file",
                            "file_name": "events.h5",
                            "file_id": "9d0637dc-97bb-418d-a833-059c2d470e02",
                            "file_path": "/ideas/toolbox/tests/data/peri_event_workflow/events.h5",
                            "file_type": "timestamp_events",
                            "file_format": "h5",
                            "file_structure": "sparse_time_series",
                            "file_category": "source",
                        },
                        {
                            "file_key": "event_aligned_traces",
                            "file_name": "event_aligned_activity.TRACES.csv",
                            "file_id": "89fef89c-1ae3-43ea-9d0b-27cf9f5560c1",
                            "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity.TRACES.csv",
                            "file_type": "event_aligned_neural_data",
                            "file_format": "csv",
                            "file_structure": "time_series",
                            "file_category": "result",
                            "parent_ids": [
                                "d5821783-5f03-4184-a502-7c68ce080107"
                            ],
                            "preview": [
                                {
                                    "name": "Mean population activity",
                                    "help": "Mean population activity over time. Shaded areas represent the different epochs. Individual events are displayed underneath the population activity trace.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/population_activity.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity across epochs",
                                    "help": "Comparison of event-aligned average population activity across the epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity",
                                    "help": "Event-aligned average population activity line plot (epoch: Baseline).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned single-cell activity figure",
                                    "help": "Event-aligned single-cell activity heatmap (epoch: Baseline)",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_single_cell_activity_heatmap_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                        {
                            "file_key": "event_aligned_statistics",
                            "file_name": "event_aligned_activity.STATISTICS.csv",
                            "file_id": "e0e970ae-1c9c-4987-b269-dc00dbc5aeca",
                            "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity.STATISTICS.csv",
                            "file_type": "statistics",
                            "file_format": "csv",
                            "file_structure": "table",
                            "file_category": "result",
                            "parent_ids": [
                                "d5821783-5f03-4184-a502-7c68ce080107",
                                "89fef89c-1ae3-43ea-9d0b-27cf9f5560c1",
                            ],
                            "preview": [
                                {
                                    "name": "Event-aligned sub-population activity figure",
                                    "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Baseline).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_by_modulation_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Spatial organization of modulation",
                                    "help": "Cell map visualizing spatial organization of modulation (epoch: Baseline).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/cell_map_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned activity of up-modulated cells",
                                    "help": "Comparison of event-aligned activity of up-modulated cells across epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_up_modulated.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned activity of down-modulated cells",
                                    "help": "Comparison of event-aligned activity of down-modulated cells across epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_down_modulated.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned activity of non-modulated cells",
                                    "help": "Comparison of event-aligned activity of non-modulated cells across epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_non_modulated.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Number of modulated cells per epoch",
                                    "help": "Number of up-, down-, and non-modulated neurons per epoch.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/num_modulated_cells_per_epoch.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Number of events per epoch",
                                    "help": "Number of events in each epoch.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_count_per_epoch.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Mean post-pre activity per epoch",
                                    "help": "Comparison of mean post-pre activity across the epochs. The error bars represent the standard error of the mean.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/mean_post_minus_pre_activity_per_epoch.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                    ],
                }
            ],
        }
        # self.assertTrue(
        #     validate_output_manifest(
        #         expected_output_manifest=exp_manifest,
        #         tool_metadata=self.tool_metadata[self.tool_key],
        #         actual_manifest_data_filepath=self.output_manifest_file,
        #     )
        # )

    def test_compare_peri_event_activity_across_epochs_series(
        self,
    ):
        """
        Validate that the compare peri-event activity across epoch tool
        can process an input consisting of a series of cell sets
        """

        # define input parameters
        epoch_names = ["Baseline", "Early Drug", "Late Drug"]
        input_parameters = {
            "event_types": ["event_type_1"],
            "visual_window": {"pre": -2, "post": 2},
            "statistical_window": {"pre": [-1, 0], "post": [0, 1]},
            "num_shuffles": 50,
            "significance_threshold": 0.25,
            "seed": 0,
            "define_epochs_by": "global file time",
            "epoch_names": ", ".join(epoch_names),
            "epoch_periods": "(10, 75), (100, 150), (300, 330)",
            "epoch_colors": "tab:grey, tab:blue, tab:cyan",
            "epoch_min_events": 1,
        }

        # execute workflow
        input_cellset_files = [
            os.path.join(self.input_dir, "cellset_series_part1-PCA-ICA.isxd"),
            os.path.join(self.input_dir, "cellset_series_part2-PCA-ICA.isxd"),
        ]
        compare_peri_event_activity_across_epochs(
            input_cellset_files=input_cellset_files,
            input_events_h5_file=self.input_events_h5_file,
            event_type=input_parameters["event_types"][0],
            visual_window_pre=input_parameters["visual_window"]["pre"],
            visual_window_post=input_parameters["visual_window"]["post"],
            statistical_window_pre_start=input_parameters[
                "statistical_window"
            ]["pre"][0],
            statistical_window_pre_end=input_parameters["statistical_window"][
                "pre"
            ][1],
            statistical_window_post_start=input_parameters[
                "statistical_window"
            ]["post"][0],
            statistical_window_post_end=input_parameters["statistical_window"][
                "post"
            ][1],
            num_shuffles=input_parameters["num_shuffles"],
            significance_threshold=input_parameters["significance_threshold"],
            seed=input_parameters["seed"],
            define_epochs_by=input_parameters["define_epochs_by"],
            epoch_names=input_parameters["epoch_names"],
            epoch_periods=input_parameters["epoch_periods"],
            epoch_colors=input_parameters["epoch_colors"],
            epoch_min_events=input_parameters["epoch_min_events"],
            output_dir=self.output_dir,
        )

        # validate existence of output files
        self.validate_existence_of_output_files(output_dir=self.output_dir)

        # validate peri-event traces csv file
        traces_csv_file = os.path.join(
            self.output_dir, "event_aligned_activity.TRACES.csv"
        )
        traces_df = pd.read_csv(traces_csv_file, header=[0, 1])
        self.assertEqual(traces_df.shape, (41, 154))
        self.validate_traces_file_column_names(
            traces_df=traces_df, epoch_names=epoch_names, cell_ids=range(20)
        )

        # validate peri-event traces file
        statistics_csv_file = os.path.join(
            self.output_dir, "event_aligned_activity.STATISTICS.csv"
        )
        statistics_df = pd.read_csv(statistics_csv_file)
        self.assertEqual(statistics_df.shape, (72, 8))
        self.validate_statistics_file_column_names(statistics_df)

        # validate epoch comparison data file
        comparison_csv_file = os.path.join(
            self.output_dir, "pairwise_epoch_comparisons.csv"
        )
        comparison_df = pd.read_csv(comparison_csv_file)
        self.assertEqual(comparison_df.shape, (3, 12))
        self.validate_pairwise_comparisons_file_column_names(comparison_df)

        # validate output manifest
        exp_manifest = {
            "schema_version": "2.0.0",
            "groups": [
                {
                    "group_key": "compare_peri_event_activity_across_epochs_output",
                    "group_type": "tool_output",
                    "group_id": "94fbbf32-25ff-4509-aec7-2b2edca122b9",
                    "series": [],
                    "files": [
                        {
                            "file_key": "input_cellset_files",
                            "file_name": "cellset_series_part1-PCA-ICA.isxd",
                            "file_id": "ed1b62b0-883b-4738-85e6-e64f6e177418",
                            "file_path": "/ideas/toolbox/tests/data/peri_event_workflow/cellset_series_part1-PCA-ICA.isxd",
                            "file_type": "cell_set",
                            "file_format": "isxd",
                            "file_structure": "binary",
                            "file_category": "source",
                        },
                        {
                            "file_key": "input_cellset_files",
                            "file_name": "cellset_series_part2-PCA-ICA.isxd",
                            "file_id": "85e6-e64f6e177418-ed1b62b0-883b-4738",
                            "file_path": "/ideas/toolbox/tests/data/peri_event_workflow/cellset_series_part2-PCA-ICA.isxd",
                            "file_type": "cell_set",
                            "file_format": "isxd",
                            "file_structure": "binary",
                            "file_category": "source",
                        },
                        {
                            "file_key": "input_events_h5_file",
                            "file_name": "events.h5",
                            "file_id": "c8e6fb09-7267-4643-937c-8eb15f9c4789",
                            "file_path": "/ideas/toolbox/tests/data/peri_event_workflow/events.h5",
                            "file_type": "timestamp_events",
                            "file_format": "h5",
                            "file_structure": "sparse_time_series",
                            "file_category": "source",
                        },
                        {
                            "file_key": "event_aligned_traces",
                            "file_name": "event_aligned_activity.TRACES.csv",
                            "file_id": "397a2acc-dd9c-4a4b-98dd-a5d4c62753d2",
                            "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity.TRACES.csv",
                            "file_type": "event_aligned_neural_data",
                            "file_format": "csv",
                            "file_structure": "time_series",
                            "file_category": "result",
                            "parent_ids": [
                                "ed1b62b0-883b-4738-85e6-e64f6e177418"
                            ],
                            "preview": [
                                {
                                    "name": "Mean population activity",
                                    "help": "Mean population activity over time. Shaded areas represent the different epochs. Individual events are displayed underneath the population activity trace.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/population_activity.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity across epochs",
                                    "help": "Comparison of event-aligned average population activity across the epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity",
                                    "help": "Event-aligned average population activity line plot (epoch: Baseline).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity",
                                    "help": "Event-aligned average population activity line plot (epoch: Early Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity_EarlyDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity",
                                    "help": "Event-aligned average population activity line plot (epoch: Late Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity_LateDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned single-cell activity figure",
                                    "help": "Event-aligned single-cell activity heatmap (epoch: Baseline)",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_single_cell_activity_heatmap_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned single-cell activity figure",
                                    "help": "Event-aligned single-cell activity heatmap (epoch: Early Drug)",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_single_cell_activity_heatmap_EarlyDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned single-cell activity figure",
                                    "help": "Event-aligned single-cell activity heatmap (epoch: Late Drug)",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_single_cell_activity_heatmap_LateDrug.preview.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                        {
                            "file_key": "event_aligned_statistics",
                            "file_name": "event_aligned_activity.STATISTICS.csv",
                            "file_id": "4b96f018-98d0-40c9-981a-1db18e20b5a2",
                            "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity.STATISTICS.csv",
                            "file_type": "statistics",
                            "file_format": "csv",
                            "file_structure": "table",
                            "file_category": "result",
                            "parent_ids": [
                                "ed1b62b0-883b-4738-85e6-e64f6e177418",
                                "397a2acc-dd9c-4a4b-98dd-a5d4c62753d2",
                            ],
                            "preview": [
                                {
                                    "name": "Event-aligned sub-population activity figure",
                                    "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Baseline).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_by_modulation_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned sub-population activity figure",
                                    "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Early Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_by_modulation_EarlyDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned sub-population activity figure",
                                    "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Late Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_by_modulation_LateDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Spatial organization of modulation",
                                    "help": "Cell map visualizing spatial organization of modulation (epoch: Baseline).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/cell_map_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Spatial organization of modulation",
                                    "help": "Cell map visualizing spatial organization of modulation (epoch: Early Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/cell_map_EarlyDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Spatial organization of modulation",
                                    "help": "Cell map visualizing spatial organization of modulation (epoch: Late Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/cell_map_LateDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned activity of up-modulated cells",
                                    "help": "Comparison of event-aligned activity of up-modulated cells across epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_up_modulated.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned activity of down-modulated cells",
                                    "help": "Comparison of event-aligned activity of down-modulated cells across epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_down_modulated.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned activity of non-modulated cells",
                                    "help": "Comparison of event-aligned activity of non-modulated cells across epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_non_modulated.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Number of modulated cells per epoch",
                                    "help": "Number of up-, down-, and non-modulated neurons per epoch.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/num_modulated_cells_per_epoch.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Number of events per epoch",
                                    "help": "Number of events in each epoch.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_count_per_epoch.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Mean post-pre activity per epoch",
                                    "help": "Comparison of mean post-pre activity across the epochs. The error bars represent the standard error of the mean.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/mean_post_minus_pre_activity_per_epoch.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                        {
                            "file_key": "event_aligned_epoch_comparison_data",
                            "file_name": "pairwise_epoch_comparisons.csv",
                            "file_id": "bfc9d1f3-7f38-4578-9549-d2eb4a1eff2a",
                            "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/pairwise_epoch_comparisons.csv",
                            "file_type": "peri_event_comparison_data",
                            "file_format": "csv",
                            "file_structure": "table",
                            "file_category": "result",
                            "parent_ids": [
                                "ed1b62b0-883b-4738-85e6-e64f6e177418",
                                "397a2acc-dd9c-4a4b-98dd-a5d4c62753d2",
                                "4b96f018-98d0-40c9-981a-1db18e20b5a2",
                            ],
                            "preview": [
                                {
                                    "name": "Post-pre differences between Baseline and Early Drug",
                                    "help": "Pairwise difference of post-pre activity between epochs Baseline and Early Drug. The left panel presents the data as a histogram. The right panel contains a cell map colored by the magnitude of the difference in post-pre activity between the epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/post_minus_pre_differences_Baseline_EarlyDrug.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Post-pre differences between Baseline and Late Drug",
                                    "help": "Pairwise difference of post-pre activity between epochs Baseline and Late Drug. The left panel presents the data as a histogram. The right panel contains a cell map colored by the magnitude of the difference in post-pre activity between the epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/post_minus_pre_differences_Baseline_LateDrug.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Post-pre differences between Early Drug and Late Drug",
                                    "help": "Pairwise difference of post-pre activity between epochs Early Drug and Late Drug. The left panel presents the data as a histogram. The right panel contains a cell map colored by the magnitude of the difference in post-pre activity between the epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/post_minus_pre_differences_EarlyDrug_LateDrug.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Post-pre distribution across the epochs",
                                    "help": "Distribution of post-pre activity across epochs displayed using a box plot. Lines connect the same cells together.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/post_minus_pre_boxplot.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                    ],
                }
            ],
        }
        # self.assertTrue(
        #     validate_output_manifest(
        #         expected_output_manifest=exp_manifest,
        #         tool_metadata=self.tool_metadata[self.tool_key],
        #         actual_manifest_data_filepath=self.output_manifest_file,
        #     )
        # )

    def test_compare_peri_event_activity_across_epochs_single_event_in_one_of_the_epochs(
        self,
    ):
        """
        Validate that the compare peri-event activity across epoch tool
        completes successfully even when an epoch contains a single event.

        Number of valid events expected for each epoch as defined in this test:
        - Baseline: 3 events
        - Early Drug: 1 event
        - Late Drug: 5 events
        """
        # define input parameters
        epoch_names = ["Baseline", "Early Drug", "Late Drug"]
        input_parameters = {
            "event_types": ["event_type_1"],
            "visual_window": {"pre": -2, "post": 2},
            "statistical_window": {"pre": [-1, 0], "post": [0, 1]},
            "num_shuffles": 50,
            "significance_threshold": 0.05,
            "seed": 0,
            "define_epochs_by": "global file time",
            "epoch_names": ", ".join(epoch_names),
            "epoch_periods": "(0, 61), (130, 145), (150, 175)",
            "epoch_colors": "tab:grey, tab:blue, tab:cyan",
            "epoch_min_events": 1,
        }

        # execute workflow
        compare_peri_event_activity_across_epochs(
            input_cellset_files=self.input_cellset_isxd_files,
            input_events_h5_file=self.input_events_h5_file,
            event_type=input_parameters["event_types"][0],
            visual_window_pre=input_parameters["visual_window"]["pre"],
            visual_window_post=input_parameters["visual_window"]["post"],
            statistical_window_pre_start=input_parameters[
                "statistical_window"
            ]["pre"][0],
            statistical_window_pre_end=input_parameters["statistical_window"][
                "pre"
            ][1],
            statistical_window_post_start=input_parameters[
                "statistical_window"
            ]["post"][0],
            statistical_window_post_end=input_parameters["statistical_window"][
                "post"
            ][1],
            num_shuffles=input_parameters["num_shuffles"],
            significance_threshold=input_parameters["significance_threshold"],
            seed=input_parameters["seed"],
            define_epochs_by=input_parameters["define_epochs_by"],
            epoch_names=input_parameters["epoch_names"],
            epoch_periods=input_parameters["epoch_periods"],
            epoch_colors=input_parameters["epoch_colors"],
            epoch_min_events=input_parameters["epoch_min_events"],
            output_dir=self.output_dir,
        )

        # validate existence of output files
        self.validate_existence_of_output_files(output_dir=self.output_dir)

        # validate peri-event traces csv file
        traces_csv_file = os.path.join(
            self.output_dir, "event_aligned_activity.TRACES.csv"
        )
        traces_df = pd.read_csv(traces_csv_file, header=[0, 1])
        self.assertEqual(traces_df.shape, (41, 154))
        self.validate_traces_file_column_names(
            traces_df=traces_df, epoch_names=epoch_names, cell_ids=range(20)
        )

        # validate peri-event statistics file
        statistics_csv_file = os.path.join(
            self.output_dir, "event_aligned_activity.STATISTICS.csv"
        )
        statistics_df = pd.read_csv(statistics_csv_file)
        self.assertEqual(statistics_df.shape, (72, 8))
        self.validate_statistics_file_column_names(statistics_df)

        # validate epoch comparison data file
        comparison_csv_file = os.path.join(
            self.output_dir, "pairwise_epoch_comparisons.csv"
        )
        comparison_df = pd.read_csv(comparison_csv_file)
        self.assertEqual(comparison_df.shape, (3, 12))
        self.validate_pairwise_comparisons_file_column_names(comparison_df)

        # # validate output manifest
        # self.assertTrue(
        #     validate_output_manifest(
        #         expected_output_manifest=self.get_expected_output_manifest(),
        #         tool_metadata=self.tool_metadata[self.tool_key],
        #         actual_manifest_data_filepath=self.output_manifest_file,
        #     )
        # )

        # validate output manifest
        exp_manifest = {
            "schema_version": "2.0.0",
            "groups": [
                {
                    "group_key": "compare_peri_event_activity_across_epochs_output",
                    "group_type": "tool_output",
                    "group_id": "af433d5d-d54a-47ca-839f-69141d819b49",
                    "series": [],
                    "files": [
                        {
                            "file_key": "input_cellset_files",
                            "file_name": "cellset_series_part1-PCA-ICA.isxd",
                            "file_id": "e4a031de-64ed-4a18-ac59-913b96029aa0",
                            "file_path": "/ideas/toolbox/tests/data/peri_event_workflow/cellset_series_part1-PCA-ICA.isxd",
                            "file_type": "cell_set",
                            "file_format": "isxd",
                            "file_structure": "binary",
                            "file_category": "source",
                        },
                        {
                            "file_key": "input_events_h5_file",
                            "file_name": "events.h5",
                            "file_id": "1bcda0f1-9d79-4448-b14a-4be4bf61009c",
                            "file_path": "/ideas/toolbox/tests/data/peri_event_workflow/events.h5",
                            "file_type": "timestamp_events",
                            "file_format": "h5",
                            "file_structure": "sparse_time_series",
                            "file_category": "source",
                        },
                        {
                            "file_key": "event_aligned_traces",
                            "file_name": "event_aligned_activity.TRACES.csv",
                            "file_id": "34db0e8d-bf11-4056-8b1a-541e3e04691d",
                            "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity.TRACES.csv",
                            "file_type": "event_aligned_neural_data",
                            "file_format": "csv",
                            "file_structure": "time_series",
                            "file_category": "result",
                            "parent_ids": [
                                "e4a031de-64ed-4a18-ac59-913b96029aa0"
                            ],
                            "preview": [
                                {
                                    "name": "Mean population activity",
                                    "help": "Mean population activity over time. Shaded areas represent the different epochs. Individual events are displayed underneath the population activity trace.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/population_activity.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity across epochs",
                                    "help": "Comparison of event-aligned average population activity across the epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity",
                                    "help": "Event-aligned average population activity line plot (epoch: Baseline).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity",
                                    "help": "Event-aligned average population activity line plot (epoch: Early Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity_EarlyDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity",
                                    "help": "Event-aligned average population activity line plot (epoch: Late Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity_LateDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned single-cell activity figure",
                                    "help": "Event-aligned single-cell activity heatmap (epoch: Baseline)",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_single_cell_activity_heatmap_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned single-cell activity figure",
                                    "help": "Event-aligned single-cell activity heatmap (epoch: Early Drug)",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_single_cell_activity_heatmap_EarlyDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned single-cell activity figure",
                                    "help": "Event-aligned single-cell activity heatmap (epoch: Late Drug)",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_single_cell_activity_heatmap_LateDrug.preview.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                        {
                            "file_key": "event_aligned_statistics",
                            "file_name": "event_aligned_activity.STATISTICS.csv",
                            "file_id": "23ed3195-e580-4ae2-a899-22a720a43536",
                            "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity.STATISTICS.csv",
                            "file_type": "statistics",
                            "file_format": "csv",
                            "file_structure": "table",
                            "file_category": "result",
                            "parent_ids": [
                                "e4a031de-64ed-4a18-ac59-913b96029aa0",
                                "34db0e8d-bf11-4056-8b1a-541e3e04691d",
                            ],
                            "preview": [
                                {
                                    "name": "Event-aligned sub-population activity figure",
                                    "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Baseline).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_by_modulation_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned sub-population activity figure",
                                    "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Early Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_by_modulation_EarlyDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned sub-population activity figure",
                                    "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Late Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_by_modulation_LateDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Spatial organization of modulation",
                                    "help": "Cell map visualizing spatial organization of modulation (epoch: Baseline).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/cell_map_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Spatial organization of modulation",
                                    "help": "Cell map visualizing spatial organization of modulation (epoch: Early Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/cell_map_EarlyDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Spatial organization of modulation",
                                    "help": "Cell map visualizing spatial organization of modulation (epoch: Late Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/cell_map_LateDrug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned activity of up-modulated cells",
                                    "help": "Comparison of event-aligned activity of up-modulated cells across epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_up_modulated.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned activity of down-modulated cells",
                                    "help": "Comparison of event-aligned activity of down-modulated cells across epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_down_modulated.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned activity of non-modulated cells",
                                    "help": "Comparison of event-aligned activity of non-modulated cells across epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_non_modulated.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Number of modulated cells per epoch",
                                    "help": "Number of up-, down-, and non-modulated neurons per epoch.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/num_modulated_cells_per_epoch.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Number of events per epoch",
                                    "help": "Number of events in each epoch.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_count_per_epoch.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Mean post-pre activity per epoch",
                                    "help": "Comparison of mean post-pre activity across the epochs. The error bars represent the standard error of the mean.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/mean_post_minus_pre_activity_per_epoch.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                        {
                            "file_key": "event_aligned_epoch_comparison_data",
                            "file_name": "pairwise_epoch_comparisons.csv",
                            "file_id": "66da28da-2159-4379-a123-37f361b05032",
                            "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/pairwise_epoch_comparisons.csv",
                            "file_type": "peri_event_comparison_data",
                            "file_format": "csv",
                            "file_structure": "table",
                            "file_category": "result",
                            "parent_ids": [
                                "e4a031de-64ed-4a18-ac59-913b96029aa0",
                                "34db0e8d-bf11-4056-8b1a-541e3e04691d",
                                "23ed3195-e580-4ae2-a899-22a720a43536",
                            ],
                            "preview": [
                                {
                                    "name": "Post-pre differences between Baseline and Early Drug",
                                    "help": "Pairwise difference of post-pre activity between epochs Baseline and Early Drug. The left panel presents the data as a histogram. The right panel contains a cell map colored by the magnitude of the difference in post-pre activity between the epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/post_minus_pre_differences_Baseline_EarlyDrug.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Post-pre differences between Baseline and Late Drug",
                                    "help": "Pairwise difference of post-pre activity between epochs Baseline and Late Drug. The left panel presents the data as a histogram. The right panel contains a cell map colored by the magnitude of the difference in post-pre activity between the epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/post_minus_pre_differences_Baseline_LateDrug.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Post-pre differences between Early Drug and Late Drug",
                                    "help": "Pairwise difference of post-pre activity between epochs Early Drug and Late Drug. The left panel presents the data as a histogram. The right panel contains a cell map colored by the magnitude of the difference in post-pre activity between the epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/post_minus_pre_differences_EarlyDrug_LateDrug.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Post-pre distribution across the epochs",
                                    "help": "Distribution of post-pre activity across epochs displayed using a box plot. Lines connect the same cells together.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/post_minus_pre_boxplot.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                    ],
                }
            ],
        }
        # self.assertTrue(
        #     validate_output_manifest(
        #         expected_output_manifest=exp_manifest,
        #         tool_metadata=self.tool_metadata[self.tool_key],
        #         actual_manifest_data_filepath=self.output_manifest_file,
        #     )
        # )

    def test_compare_peri_event_activity_across_epochs_single_event_in_all_epochs(
        self,
    ):
        """
        Validate that the compare peri-event activity across epoch tool
        completes successfully when each epoch contains a single valid event.

        Number of valid events expected for each epoch as defined in this test:
        - Baseline: 1 event
        - Drug: 1 event
        """
        # define input parameters
        epoch_names = ["Baseline", "Drug"]
        input_parameters = {
            "event_types": ["event_type_1"],
            "visual_window": {"pre": -2, "post": 2},
            "statistical_window": {"pre": [-1, 0], "post": [0, 1]},
            "num_shuffles": 50,
            "significance_threshold": 0.05,
            "seed": 0,
            "define_epochs_by": "global file time",
            "epoch_names": ", ".join(epoch_names),
            "epoch_periods": "(50, 61), (130, 145)",
            "epoch_colors": "tab:grey, tab:blue",
            "epoch_min_events": 1,
        }

        # execute workflow
        compare_peri_event_activity_across_epochs(
            input_cellset_files=self.input_cellset_isxd_files,
            input_events_h5_file=self.input_events_h5_file,
            event_type=input_parameters["event_types"][0],
            visual_window_pre=input_parameters["visual_window"]["pre"],
            visual_window_post=input_parameters["visual_window"]["post"],
            statistical_window_pre_start=input_parameters[
                "statistical_window"
            ]["pre"][0],
            statistical_window_pre_end=input_parameters["statistical_window"][
                "pre"
            ][1],
            statistical_window_post_start=input_parameters[
                "statistical_window"
            ]["post"][0],
            statistical_window_post_end=input_parameters["statistical_window"][
                "post"
            ][1],
            num_shuffles=input_parameters["num_shuffles"],
            significance_threshold=input_parameters["significance_threshold"],
            seed=input_parameters["seed"],
            define_epochs_by=input_parameters["define_epochs_by"],
            epoch_names=input_parameters["epoch_names"],
            epoch_periods=input_parameters["epoch_periods"],
            epoch_colors=input_parameters["epoch_colors"],
            epoch_min_events=input_parameters["epoch_min_events"],
            output_dir=self.output_dir,
        )

        # validate existence of output files
        act_output_files = os.listdir(self.output_dir)
        for f in [
            # event-aligned traces
            "event_aligned_activity.TRACES.csv",
            "population_activity.preview.svg",
            "event_aligned_population_activity.preview.svg",
            "event_aligned_population_activity_Baseline.preview.svg",
            "event_aligned_population_activity_Drug.preview.svg",
            "event_aligned_single_cell_activity_heatmap_Baseline.preview.svg",
            "event_aligned_single_cell_activity_heatmap_Drug.preview.svg",
            # event-aligned statistics
            "event_aligned_activity.STATISTICS.csv",
            "event_aligned_activity_by_modulation_Baseline.preview.svg",
            "event_aligned_activity_by_modulation_Drug.preview.svg",
            "cell_map_Baseline.preview.svg",
            "cell_map_Drug.preview.svg",
            # "event_aligned_activity_up_modulated.svg", --> no up-modulated cells
            # "event_aligned_activity_down_modulated.svg", --> no down-modulated cells
            "event_aligned_activity_non_modulated.svg",
            "event_count_per_epoch.preview.svg",
            "num_modulated_cells_per_epoch.svg",
            "mean_post_minus_pre_activity_per_epoch.svg",
            # event-aligned epoch comparison data
            "pairwise_epoch_comparisons.csv",
            "post_minus_pre_differences_Baseline_Drug.svg",
            "post_minus_pre_boxplot.svg",
            # output manifest & metadata
            # "output_manifest.json",
            # "output_metadata.json",
        ]:
            self.assertTrue(f in act_output_files)

        # validate peri-event traces csv file
        traces_csv_file = os.path.join(
            self.output_dir, "event_aligned_activity.TRACES.csv"
        )
        traces_df = pd.read_csv(traces_csv_file, header=[0, 1])
        self.assertEqual(traces_df.shape, (41, 103))
        self.validate_traces_file_column_names(
            traces_df=traces_df, epoch_names=epoch_names, cell_ids=range(20)
        )

        # validate peri-event statistics file
        statistics_csv_file = os.path.join(
            self.output_dir, "event_aligned_activity.STATISTICS.csv"
        )
        statistics_df = pd.read_csv(statistics_csv_file)
        self.assertEqual(statistics_df.shape, (48, 8))
        self.validate_statistics_file_column_names(statistics_df)

        # validate epoch comparison data file
        comparison_csv_file = os.path.join(
            self.output_dir, "pairwise_epoch_comparisons.csv"
        )
        comparison_df = pd.read_csv(comparison_csv_file)
        self.assertEqual(comparison_df.shape, (1, 12))
        self.validate_pairwise_comparisons_file_column_names(comparison_df)

        # validate output manifest
        exp_manifest = {
            "schema_version": "2.0.0",
            "groups": [
                {
                    "group_key": "compare_peri_event_activity_across_epochs_output",
                    "group_type": "tool_output",
                    "group_id": "3c344fea-55c1-48f0-bf22-acfd057cf8ef",
                    "series": [],
                    "files": [
                        {
                            "file_key": "input_cellset_files",
                            "file_name": "cellset_series_part1-PCA-ICA.isxd",
                            "file_id": "c62db136-d662-4d1c-bcf4-a980cc288f45",
                            "file_path": "/ideas/toolbox/tests/data/peri_event_workflow/cellset_series_part1-PCA-ICA.isxd",
                            "file_type": "cell_set",
                            "file_format": "isxd",
                            "file_structure": "binary",
                            "file_category": "source",
                        },
                        {
                            "file_key": "input_events_h5_file",
                            "file_name": "events.h5",
                            "file_id": "6e3346ab-2f9e-4e05-954f-da79d7f664ef",
                            "file_path": "/ideas/toolbox/tests/data/peri_event_workflow/events.h5",
                            "file_type": "timestamp_events",
                            "file_format": "h5",
                            "file_structure": "sparse_time_series",
                            "file_category": "source",
                        },
                        {
                            "file_key": "event_aligned_traces",
                            "file_name": "event_aligned_activity.TRACES.csv",
                            "file_id": "4314890b-ea7b-4da0-920c-a70d69a30d6c",
                            "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity.TRACES.csv",
                            "file_type": "event_aligned_neural_data",
                            "file_format": "csv",
                            "file_structure": "time_series",
                            "file_category": "result",
                            "parent_ids": [
                                "c62db136-d662-4d1c-bcf4-a980cc288f45"
                            ],
                            "preview": [
                                {
                                    "name": "Mean population activity",
                                    "help": "Mean population activity over time. Shaded areas represent the different epochs. Individual events are displayed underneath the population activity trace.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/population_activity.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity across epochs",
                                    "help": "Comparison of event-aligned average population activity across the epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity",
                                    "help": "Event-aligned average population activity line plot (epoch: Baseline).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity",
                                    "help": "Event-aligned average population activity line plot (epoch: Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity_Drug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned single-cell activity figure",
                                    "help": "Event-aligned single-cell activity heatmap (epoch: Baseline)",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_single_cell_activity_heatmap_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned single-cell activity figure",
                                    "help": "Event-aligned single-cell activity heatmap (epoch: Drug)",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_single_cell_activity_heatmap_Drug.preview.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                        {
                            "file_key": "event_aligned_statistics",
                            "file_name": "event_aligned_activity.STATISTICS.csv",
                            "file_id": "4887e4c3-abc3-42b2-a901-ff967bb12fee",
                            "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity.STATISTICS.csv",
                            "file_type": "statistics",
                            "file_format": "csv",
                            "file_structure": "table",
                            "file_category": "result",
                            "parent_ids": [
                                "c62db136-d662-4d1c-bcf4-a980cc288f45",
                                "4314890b-ea7b-4da0-920c-a70d69a30d6c",
                            ],
                            "preview": [
                                {
                                    "name": "Event-aligned sub-population activity figure",
                                    "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Baseline).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_by_modulation_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned sub-population activity figure",
                                    "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_by_modulation_Drug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Spatial organization of modulation",
                                    "help": "Cell map visualizing spatial organization of modulation (epoch: Baseline).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/cell_map_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Spatial organization of modulation",
                                    "help": "Cell map visualizing spatial organization of modulation (epoch: Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/cell_map_Drug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned activity of down-modulated cells",
                                    "help": "Comparison of event-aligned activity of down-modulated cells across epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_down_modulated.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned activity of non-modulated cells",
                                    "help": "Comparison of event-aligned activity of non-modulated cells across epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_non_modulated.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Number of modulated cells per epoch",
                                    "help": "Number of up-, down-, and non-modulated neurons per epoch.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/num_modulated_cells_per_epoch.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Number of events per epoch",
                                    "help": "Number of events in each epoch.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_count_per_epoch.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Mean post-pre activity per epoch",
                                    "help": "Comparison of mean post-pre activity across the epochs. The error bars represent the standard error of the mean.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/mean_post_minus_pre_activity_per_epoch.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                        {
                            "file_key": "event_aligned_epoch_comparison_data",
                            "file_name": "pairwise_epoch_comparisons.csv",
                            "file_id": "cd6c85b2-2ca6-4d7f-adbb-fd6e2b690ac1",
                            "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/pairwise_epoch_comparisons.csv",
                            "file_type": "peri_event_comparison_data",
                            "file_format": "csv",
                            "file_structure": "table",
                            "file_category": "result",
                            "parent_ids": [
                                "c62db136-d662-4d1c-bcf4-a980cc288f45",
                                "4314890b-ea7b-4da0-920c-a70d69a30d6c",
                                "4887e4c3-abc3-42b2-a901-ff967bb12fee",
                            ],
                            "preview": [
                                {
                                    "name": "Post-pre differences between Baseline and Drug",
                                    "help": "Pairwise difference of post-pre activity between epochs Baseline and Drug. The left panel presents the data as a histogram. The right panel contains a cell map colored by the magnitude of the difference in post-pre activity between the epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/post_minus_pre_differences_Baseline_Drug.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Post-pre distribution across the epochs",
                                    "help": "Distribution of post-pre activity across epochs displayed using a box plot. Lines connect the same cells together.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/post_minus_pre_boxplot.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                    ],
                }
            ],
        }
        # self.assertTrue(
        #     validate_output_manifest(
        #         expected_output_manifest=exp_manifest,
        #         tool_metadata=self.tool_metadata[self.tool_key],
        #         actual_manifest_data_filepath=self.output_manifest_file,
        #     )
        # )

    def test_compare_peri_event_activity_across_epochs_define_epoch_by_file(
        self,
    ):
        """
        Validate that the compare peri-event activity across epoch tool
        can process data correctly when defining epochs by file.
        """

        # define input parameters
        epoch_names = ["Baseline", "Drug"]
        input_parameters = {
            "event_types": ["event_type_1"],
            "visual_window": {"pre": -2, "post": 2},
            "statistical_window": {"pre": [-1, 0], "post": [0, 1]},
            "num_shuffles": 50,
            "significance_threshold": 0.05,
            "seed": 0,
            "define_epochs_by": "files",
            "epoch_names": ", ".join(epoch_names),
            "epoch_periods": "will be automatically updated",
            "epoch_colors": "tab:grey, tab:blue",
            "epoch_min_events": 1,
        }

        # execute workflow
        input_cellset_files = [
            os.path.join(self.input_dir, "cellset_series_part1-PCA-ICA.isxd"),
            os.path.join(self.input_dir, "cellset_series_part2-PCA-ICA.isxd"),
        ]
        compare_peri_event_activity_across_epochs(
            input_cellset_files=input_cellset_files,
            input_events_h5_file=self.input_events_h5_file,
            event_type=input_parameters["event_types"][0],
            visual_window_pre=input_parameters["visual_window"]["pre"],
            visual_window_post=input_parameters["visual_window"]["post"],
            statistical_window_pre_start=input_parameters[
                "statistical_window"
            ]["pre"][0],
            statistical_window_pre_end=input_parameters["statistical_window"][
                "pre"
            ][1],
            statistical_window_post_start=input_parameters[
                "statistical_window"
            ]["post"][0],
            statistical_window_post_end=input_parameters["statistical_window"][
                "post"
            ][1],
            num_shuffles=input_parameters["num_shuffles"],
            significance_threshold=input_parameters["significance_threshold"],
            seed=input_parameters["seed"],
            define_epochs_by=input_parameters["define_epochs_by"],
            epoch_names=input_parameters["epoch_names"],
            epoch_periods=input_parameters["epoch_periods"],
            epoch_colors=input_parameters["epoch_colors"],
            epoch_min_events=input_parameters["epoch_min_events"],
            output_dir=self.output_dir,
        )

        # validate existence of output files
        act_output_files = os.listdir(self.output_dir)
        for f in [
            # event-aligned traces
            "event_aligned_activity.TRACES.csv",
            "population_activity.preview.svg",
            "event_aligned_population_activity.preview.svg",
            "event_aligned_population_activity_Baseline.preview.svg",
            "event_aligned_population_activity_Drug.preview.svg",
            "event_aligned_single_cell_activity_heatmap_Baseline.preview.svg",
            "event_aligned_single_cell_activity_heatmap_Drug.preview.svg",
            # event-aligned statistics
            "event_aligned_activity.STATISTICS.csv",
            "event_aligned_activity_by_modulation_Baseline.preview.svg",
            "event_aligned_activity_by_modulation_Drug.preview.svg",
            "cell_map_Baseline.preview.svg",
            "cell_map_Drug.preview.svg",
            # "event_aligned_activity_up_modulated.svg", --> no up-modulated cells
            "event_aligned_activity_down_modulated.svg",
            "event_aligned_activity_non_modulated.svg",
            "event_count_per_epoch.preview.svg",
            "num_modulated_cells_per_epoch.svg",
            "mean_post_minus_pre_activity_per_epoch.svg",
            # event-aligned epoch comparison data
            "pairwise_epoch_comparisons.csv",
            "post_minus_pre_differences_Baseline_Drug.svg",
            "post_minus_pre_boxplot.svg",
            # output manifest & metadata
            # "output_manifest.json",
            # "output_metadata.json",
        ]:
            self.assertTrue(f in act_output_files)

        # validate peri-event traces csv file
        traces_csv_file = os.path.join(
            self.output_dir, "event_aligned_activity.TRACES.csv"
        )
        traces_df = pd.read_csv(traces_csv_file, header=[0, 1])
        self.assertEqual(traces_df.shape, (41, 103))
        self.validate_traces_file_column_names(
            traces_df=traces_df, epoch_names=epoch_names, cell_ids=range(20)
        )

        # validate peri-event traces file
        statistics_csv_file = os.path.join(
            self.output_dir, "event_aligned_activity.STATISTICS.csv"
        )
        statistics_df = pd.read_csv(statistics_csv_file)
        self.assertEqual(statistics_df.shape, (48, 8))
        self.validate_statistics_file_column_names(statistics_df)

        # validate epoch comparison data file
        comparison_csv_file = os.path.join(
            self.output_dir, "pairwise_epoch_comparisons.csv"
        )
        comparison_df = pd.read_csv(comparison_csv_file)
        self.assertEqual(comparison_df.shape, (1, 12))
        self.validate_pairwise_comparisons_file_column_names(comparison_df)

        # validate output manifest
        exp_manifest = {
            "schema_version": "2.0.0",
            "groups": [
                {
                    "group_key": "compare_peri_event_activity_across_epochs_output",
                    "group_type": "tool_output",
                    "group_id": "6525ce5b-9d96-45a2-bdd0-a89731551774",
                    "series": [],
                    "files": [
                        {
                            "file_key": "input_cellset_files",
                            "file_name": "cellset_series_part1-PCA-ICA.isxd",
                            "file_id": "e2e66b12-8890-48fc-9a83-8e6b929540a6",
                            "file_path": "/ideas/toolbox/tests/data/peri_event_workflow/cellset_series_part1-PCA-ICA.isxd",
                            "file_type": "cell_set",
                            "file_format": "isxd",
                            "file_structure": "binary",
                            "file_category": "source",
                        },
                        {
                            "file_key": "input_cellset_files",
                            "file_name": "cellset_series_part2-PCA-ICA.isxd",
                            "file_id": "8e54de6e-b478-4d0d-af7f-5b12f9ef8053",
                            "file_path": "/ideas/toolbox/tests/data/peri_event_workflow/cellset_series_part2-PCA-ICA.isxd",
                            "file_type": "cell_set",
                            "file_format": "isxd",
                            "file_structure": "binary",
                            "file_category": "source",
                        },
                        {
                            "file_key": "input_events_h5_file",
                            "file_name": "events.h5",
                            "file_id": "32152b0a-161b-488e-b05e-88a0659f98f7",
                            "file_path": "/ideas/toolbox/tests/data/peri_event_workflow/events.h5",
                            "file_type": "timestamp_events",
                            "file_format": "h5",
                            "file_structure": "sparse_time_series",
                            "file_category": "source",
                        },
                        {
                            "file_key": "event_aligned_traces",
                            "file_name": "event_aligned_activity.TRACES.csv",
                            "file_id": "714ceec6-0e1e-4dc4-a143-6d80399fed09",
                            "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity.TRACES.csv",
                            "file_type": "event_aligned_neural_data",
                            "file_format": "csv",
                            "file_structure": "time_series",
                            "file_category": "result",
                            "parent_ids": [
                                "e2e66b12-8890-48fc-9a83-8e6b929540a6",
                                "8e54de6e-b478-4d0d-af7f-5b12f9ef8053",
                            ],
                            "preview": [
                                {
                                    "name": "Mean population activity",
                                    "help": "Mean population activity over time. Shaded areas represent the different epochs. Individual events are displayed underneath the population activity trace.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/population_activity.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity across epochs",
                                    "help": "Comparison of event-aligned average population activity across the epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity",
                                    "help": "Event-aligned average population activity line plot (epoch: Baseline).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity",
                                    "help": "Event-aligned average population activity line plot (epoch: Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity_Drug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned single-cell activity figure",
                                    "help": "Event-aligned single-cell activity heatmap (epoch: Baseline)",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_single_cell_activity_heatmap_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned single-cell activity figure",
                                    "help": "Event-aligned single-cell activity heatmap (epoch: Drug)",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_single_cell_activity_heatmap_Drug.preview.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                        {
                            "file_key": "event_aligned_statistics",
                            "file_name": "event_aligned_activity.STATISTICS.csv",
                            "file_id": "c06570d2-9c91-42d6-8059-39264e105858",
                            "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity.STATISTICS.csv",
                            "file_type": "statistics",
                            "file_format": "csv",
                            "file_structure": "table",
                            "file_category": "result",
                            "parent_ids": [
                                "e2e66b12-8890-48fc-9a83-8e6b929540a6",
                                "714ceec6-0e1e-4dc4-a143-6d80399fed09",
                            ],
                            "preview": [
                                {
                                    "name": "Event-aligned sub-population activity figure",
                                    "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Baseline).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_by_modulation_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned sub-population activity figure",
                                    "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_by_modulation_Drug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Spatial organization of modulation",
                                    "help": "Cell map visualizing spatial organization of modulation (epoch: Baseline).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/cell_map_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Spatial organization of modulation",
                                    "help": "Cell map visualizing spatial organization of modulation (epoch: Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/cell_map_Drug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned activity of down-modulated cells",
                                    "help": "Comparison of event-aligned activity of down-modulated cells across epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_down_modulated.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned activity of non-modulated cells",
                                    "help": "Comparison of event-aligned activity of non-modulated cells across epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_non_modulated.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Number of modulated cells per epoch",
                                    "help": "Number of up-, down-, and non-modulated neurons per epoch.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/num_modulated_cells_per_epoch.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Number of events per epoch",
                                    "help": "Number of events in each epoch.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_count_per_epoch.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Mean post-pre activity per epoch",
                                    "help": "Comparison of mean post-pre activity across the epochs. The error bars represent the standard error of the mean.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/mean_post_minus_pre_activity_per_epoch.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                        {
                            "file_key": "event_aligned_epoch_comparison_data",
                            "file_name": "pairwise_epoch_comparisons.csv",
                            "file_id": "948725be-13a7-4873-abd3-f5c1c5abb78b",
                            "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/pairwise_epoch_comparisons.csv",
                            "file_type": "peri_event_comparison_data",
                            "file_format": "csv",
                            "file_structure": "table",
                            "file_category": "result",
                            "parent_ids": [
                                "e2e66b12-8890-48fc-9a83-8e6b929540a6",
                                "714ceec6-0e1e-4dc4-a143-6d80399fed09",
                                "c06570d2-9c91-42d6-8059-39264e105858",
                            ],
                            "preview": [
                                {
                                    "name": "Post-pre differences between Baseline and Drug",
                                    "help": "Pairwise difference of post-pre activity between epochs Baseline and Drug. The left panel presents the data as a histogram. The right panel contains a cell map colored by the magnitude of the difference in post-pre activity between the epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/post_minus_pre_differences_Baseline_Drug.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Post-pre distribution across the epochs",
                                    "help": "Distribution of post-pre activity across epochs displayed using a box plot. Lines connect the same cells together.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/post_minus_pre_boxplot.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                    ],
                }
            ],
        }
        # self.assertTrue(
        #     validate_output_manifest(
        #         expected_output_manifest=exp_manifest,
        #         tool_metadata=self.tool_metadata[self.tool_key],
        #         actual_manifest_data_filepath=self.output_manifest_file,
        #     )
        # )

    def test_compare_peri_event_activity_across_epochs_define_epochs_by_local_file_time(
        self,
    ):
        """
        Validate that the compare peri-event activity across epoch tool
        can process data correctly when defining epochs by local file time.
        """

        # define input parameters
        epoch_names = ["Baseline", "Drug"]
        input_parameters = {
            "event_types": ["event_type_1"],
            "visual_window": {"pre": -2, "post": 2},
            "statistical_window": {"pre": [-1, 0], "post": [0, 1]},
            "num_shuffles": 50,
            "significance_threshold": 0.05,
            "seed": 0,
            "define_epochs_by": "local file time",
            "epoch_names": ", ".join(epoch_names),
            "epoch_periods": "(10, 60), (50, 109)",
            "epoch_colors": "tab:grey, tab:blue",
            "epoch_min_events": 1,
        }

        # execute workflow
        input_cellset_files = [
            os.path.join(self.input_dir, "cellset_series_part1-PCA-ICA.isxd"),
            os.path.join(self.input_dir, "cellset_series_part2-PCA-ICA.isxd"),
        ]
        compare_peri_event_activity_across_epochs(
            input_cellset_files=input_cellset_files,
            input_events_h5_file=self.input_events_h5_file,
            event_type=input_parameters["event_types"][0],
            visual_window_pre=input_parameters["visual_window"]["pre"],
            visual_window_post=input_parameters["visual_window"]["post"],
            statistical_window_pre_start=input_parameters[
                "statistical_window"
            ]["pre"][0],
            statistical_window_pre_end=input_parameters["statistical_window"][
                "pre"
            ][1],
            statistical_window_post_start=input_parameters[
                "statistical_window"
            ]["post"][0],
            statistical_window_post_end=input_parameters["statistical_window"][
                "post"
            ][1],
            num_shuffles=input_parameters["num_shuffles"],
            significance_threshold=input_parameters["significance_threshold"],
            seed=input_parameters["seed"],
            define_epochs_by=input_parameters["define_epochs_by"],
            epoch_names=input_parameters["epoch_names"],
            epoch_periods=input_parameters["epoch_periods"],
            epoch_colors=input_parameters["epoch_colors"],
            epoch_min_events=input_parameters["epoch_min_events"],
            output_dir=self.output_dir,
        )

        # validate existence of output files
        act_output_files = os.listdir(self.output_dir)
        for f in [
            # event-aligned traces
            "event_aligned_activity.TRACES.csv",
            "population_activity.preview.svg",
            "event_aligned_population_activity.preview.svg",
            "event_aligned_population_activity_Baseline.preview.svg",
            "event_aligned_population_activity_Drug.preview.svg",
            "event_aligned_single_cell_activity_heatmap_Baseline.preview.svg",
            "event_aligned_single_cell_activity_heatmap_Drug.preview.svg",
            # event-aligned statistics
            "event_aligned_activity.STATISTICS.csv",
            "event_aligned_activity_by_modulation_Baseline.preview.svg",
            "event_aligned_activity_by_modulation_Drug.preview.svg",
            "cell_map_Baseline.preview.svg",
            "cell_map_Drug.preview.svg",
            "event_aligned_activity_up_modulated.svg",
            "event_aligned_activity_down_modulated.svg",
            "event_aligned_activity_non_modulated.svg",
            "event_count_per_epoch.preview.svg",
            "num_modulated_cells_per_epoch.svg",
            "mean_post_minus_pre_activity_per_epoch.svg",
            # event-aligned epoch comparison data
            "pairwise_epoch_comparisons.csv",
            "post_minus_pre_differences_Baseline_Drug.svg",
            "post_minus_pre_boxplot.svg",
            # output manifest & metadata
            # "output_manifest.json",
            # "output_metadata.json",
        ]:
            self.assertTrue(f in act_output_files)

        # validate peri-event traces csv file
        traces_csv_file = os.path.join(
            self.output_dir, "event_aligned_activity.TRACES.csv"
        )
        traces_df = pd.read_csv(traces_csv_file, header=[0, 1])
        self.assertEqual(traces_df.shape, (41, 103))
        self.validate_traces_file_column_names(
            traces_df=traces_df, epoch_names=epoch_names, cell_ids=range(20)
        )

        # validate peri-event traces file
        statistics_csv_file = os.path.join(
            self.output_dir, "event_aligned_activity.STATISTICS.csv"
        )
        statistics_df = pd.read_csv(statistics_csv_file)
        self.assertEqual(statistics_df.shape, (48, 8))
        self.validate_statistics_file_column_names(statistics_df)

        # validate epoch comparison data file
        comparison_csv_file = os.path.join(
            self.output_dir, "pairwise_epoch_comparisons.csv"
        )
        comparison_df = pd.read_csv(comparison_csv_file)
        self.assertEqual(comparison_df.shape, (1, 12))
        self.validate_pairwise_comparisons_file_column_names(comparison_df)

        # validate output manifest
        exp_manifest = {
            "schema_version": "2.0.0",
            "groups": [
                {
                    "group_key": "compare_peri_event_activity_across_epochs_output",
                    "group_type": "tool_output",
                    "group_id": "fec4d376-a62b-4c74-bd3f-4316d311bbf9",
                    "series": [],
                    "files": [
                        {
                            "file_key": "input_cellset_files",
                            "file_name": "cellset_series_part1-PCA-ICA.isxd",
                            "file_id": "64b6c5ae-e358-4666-b67b-f2899dd6e54c",
                            "file_path": "/ideas/toolbox/tests/data/peri_event_workflow/cellset_series_part1-PCA-ICA.isxd",
                            "file_type": "cell_set",
                            "file_format": "isxd",
                            "file_structure": "binary",
                            "file_category": "source",
                        },
                        {
                            "file_key": "input_cellset_files",
                            "file_name": "cellset_series_part2-PCA-ICA.isxd",
                            "file_id": "5bd1e4d6-a5e3-431d-838b-114d026761cb",
                            "file_path": "/ideas/toolbox/tests/data/peri_event_workflow/cellset_series_part2-PCA-ICA.isxd",
                            "file_type": "cell_set",
                            "file_format": "isxd",
                            "file_structure": "binary",
                            "file_category": "source",
                        },
                        {
                            "file_key": "input_events_h5_file",
                            "file_name": "events.h5",
                            "file_id": "56f76353-1fe7-4115-b9ff-c20d4cfb21c7",
                            "file_path": "/ideas/toolbox/tests/data/peri_event_workflow/events.h5",
                            "file_type": "timestamp_events",
                            "file_format": "h5",
                            "file_structure": "sparse_time_series",
                            "file_category": "source",
                        },
                        {
                            "file_key": "event_aligned_traces",
                            "file_name": "event_aligned_activity.TRACES.csv",
                            "file_id": "ff9fc76c-6e17-4d1a-b7da-cb0aa12b9362",
                            "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity.TRACES.csv",
                            "file_type": "event_aligned_neural_data",
                            "file_format": "csv",
                            "file_structure": "time_series",
                            "file_category": "result",
                            "parent_ids": [
                                "64b6c5ae-e358-4666-b67b-f2899dd6e54c",
                                "5bd1e4d6-a5e3-431d-838b-114d026761cb",
                            ],
                            "preview": [
                                {
                                    "name": "Mean population activity",
                                    "help": "Mean population activity over time. Shaded areas represent the different epochs. Individual events are displayed underneath the population activity trace.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/population_activity.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity across epochs",
                                    "help": "Comparison of event-aligned average population activity across the epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity",
                                    "help": "Event-aligned average population activity line plot (epoch: Baseline).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned population activity",
                                    "help": "Event-aligned average population activity line plot (epoch: Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_population_activity_Drug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned single-cell activity figure",
                                    "help": "Event-aligned single-cell activity heatmap (epoch: Baseline)",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_single_cell_activity_heatmap_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned single-cell activity figure",
                                    "help": "Event-aligned single-cell activity heatmap (epoch: Drug)",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_single_cell_activity_heatmap_Drug.preview.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                        {
                            "file_key": "event_aligned_statistics",
                            "file_name": "event_aligned_activity.STATISTICS.csv",
                            "file_id": "40b82b81-bd2e-4eb6-878c-e1801d811017",
                            "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity.STATISTICS.csv",
                            "file_type": "statistics",
                            "file_format": "csv",
                            "file_structure": "table",
                            "file_category": "result",
                            "parent_ids": [
                                "64b6c5ae-e358-4666-b67b-f2899dd6e54c",
                                "ff9fc76c-6e17-4d1a-b7da-cb0aa12b9362",
                            ],
                            "preview": [
                                {
                                    "name": "Event-aligned sub-population activity figure",
                                    "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Baseline).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_by_modulation_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned sub-population activity figure",
                                    "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_by_modulation_Drug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Spatial organization of modulation",
                                    "help": "Cell map visualizing spatial organization of modulation (epoch: Baseline).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/cell_map_Baseline.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Spatial organization of modulation",
                                    "help": "Cell map visualizing spatial organization of modulation (epoch: Drug).",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/cell_map_Drug.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned activity of up-modulated cells",
                                    "help": "Comparison of event-aligned activity of up-modulated cells across epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_up_modulated.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned activity of down-modulated cells",
                                    "help": "Comparison of event-aligned activity of down-modulated cells across epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_down_modulated.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned activity of non-modulated cells",
                                    "help": "Comparison of event-aligned activity of non-modulated cells across epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_aligned_activity_non_modulated.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Number of modulated cells per epoch",
                                    "help": "Number of up-, down-, and non-modulated neurons per epoch.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/num_modulated_cells_per_epoch.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Number of events per epoch",
                                    "help": "Number of events in each epoch.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/event_count_per_epoch.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Mean post-pre activity per epoch",
                                    "help": "Comparison of mean post-pre activity across the epochs. The error bars represent the standard error of the mean.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/mean_post_minus_pre_activity_per_epoch.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                        {
                            "file_key": "event_aligned_epoch_comparison_data",
                            "file_name": "pairwise_epoch_comparisons.csv",
                            "file_id": "19d56fb0-9498-44a2-9df8-e65bbe306ce9",
                            "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/pairwise_epoch_comparisons.csv",
                            "file_type": "peri_event_comparison_data",
                            "file_format": "csv",
                            "file_structure": "table",
                            "file_category": "result",
                            "parent_ids": [
                                "64b6c5ae-e358-4666-b67b-f2899dd6e54c",
                                "ff9fc76c-6e17-4d1a-b7da-cb0aa12b9362",
                                "40b82b81-bd2e-4eb6-878c-e1801d811017",
                            ],
                            "preview": [
                                {
                                    "name": "Post-pre differences between Baseline and Drug",
                                    "help": "Pairwise difference of post-pre activity between epochs Baseline and Drug. The left panel presents the data as a histogram. The right panel contains a cell map colored by the magnitude of the difference in post-pre activity between the epochs.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/post_minus_pre_differences_Baseline_Drug.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Post-pre distribution across the epochs",
                                    "help": "Distribution of post-pre activity across epochs displayed using a box plot. Lines connect the same cells together.",
                                    "file_path": "/tmp/tmp_compare_peri_event_activity_across_epochs_outputs/post_minus_pre_boxplot.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                    ],
                }
            ],
        }
        # self.assertTrue(
        #     validate_output_manifest(
        #         expected_output_manifest=exp_manifest,
        #         tool_metadata=self.tool_metadata[self.tool_key],
        #         actual_manifest_data_filepath=self.output_manifest_file,
        #     )
        # )
