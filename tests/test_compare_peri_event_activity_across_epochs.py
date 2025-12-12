import os
import shutil
import unittest

import numpy as np
import pandas as pd
from ideas.exceptions import IdeasError

from analysis.compare_peri_event_activity_across_epochs import (
    compare_peri_event_activity_across_epochs,
)


class TestComparePeriEventActivityAcrossEpochs(unittest.TestCase):
    """Tests for the compare peri-event activity across epochs tool."""

    # tool info
    tool_key = "compare_peri_event_activity_across_epochs"

    # define directories
    temporary_dir = "/tmp"
    data_dir = "data"
    input_dir = "data/peri_event_workflow"
    output_dir = os.path.join(
        temporary_dir, "tmp_compare_peri_event_activity_across_epochs_outputs"
    )

    # output manifest
    output_manifest_json_schema = "toolbox/tests/schemas/output_manifest_schema.json"
    output_manifest_file_basename = "output_manifest.json"
    output_manifest_file = os.path.join(output_dir, output_manifest_file_basename)

    # output metadata
    output_metadata_json_schema = "toolbox/tests/schemas/output_metadata_schema.json"
    output_metadata_file_basename = "output_metadata.json"
    output_metadata_file = os.path.join(output_dir, output_metadata_file_basename)

    # input files
    input_cellset_isxd_files = [
        os.path.join(data_dir, "cellset_series_part1-PCA-ICA.isxd")
    ]
    input_events_h5_file = os.path.join(data_dir, "events.h5")

    def setUp(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def validate_existence_of_output_files(self, output_dir, files_to_exclude=[]):
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
            "output_metadata.json",
        ]:
            if f not in files_to_exclude:
                self.assertTrue(f in output_files)

    def validate_traces_file_column_names(self, traces_df, epoch_names, cell_ids):
        """Validate output traces file column names."""
        exp_epoch_headers = epoch_names + ["Time"]
        act_epochs_headers = np.unique(traces_df.columns.get_level_values(0)).tolist()
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
                self.assertTrue((epoch_name, second_level_header) in traces_df.columns)

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

    def validate_pairwise_comparisons_file_column_names(self, df, cols_to_exclude=None):
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
            statistical_window_pre_start=input_parameters["statistical_window"]["pre"][
                0
            ],
            statistical_window_pre_end=input_parameters["statistical_window"]["pre"][1],
            statistical_window_post_start=input_parameters["statistical_window"][
                "post"
            ][0],
            statistical_window_post_end=input_parameters["statistical_window"]["post"][
                1
            ],
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
            statistical_window_pre_start=input_parameters["statistical_window"]["pre"][
                0
            ],
            statistical_window_pre_end=input_parameters["statistical_window"]["pre"][1],
            statistical_window_post_start=input_parameters["statistical_window"][
                "post"
            ][0],
            statistical_window_post_end=input_parameters["statistical_window"]["post"][
                1
            ],
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
            statistical_window_pre_start=input_parameters["statistical_window"]["pre"][
                0
            ],
            statistical_window_pre_end=input_parameters["statistical_window"]["pre"][1],
            statistical_window_post_start=input_parameters["statistical_window"][
                "post"
            ][0],
            statistical_window_post_end=input_parameters["statistical_window"]["post"][
                1
            ],
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
            statistical_window_pre_start=input_parameters["statistical_window"]["pre"][
                0
            ],
            statistical_window_pre_end=input_parameters["statistical_window"]["pre"][1],
            statistical_window_post_start=input_parameters["statistical_window"][
                "post"
            ][0],
            statistical_window_post_end=input_parameters["statistical_window"]["post"][
                1
            ],
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
            statistical_window_pre_start=input_parameters["statistical_window"]["pre"][
                0
            ],
            statistical_window_pre_end=input_parameters["statistical_window"]["pre"][1],
            statistical_window_post_start=input_parameters["statistical_window"][
                "post"
            ][0],
            statistical_window_post_end=input_parameters["statistical_window"]["post"][
                1
            ],
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
            statistical_window_pre_start=input_parameters["statistical_window"]["pre"][
                0
            ],
            statistical_window_pre_end=input_parameters["statistical_window"]["pre"][1],
            statistical_window_post_start=input_parameters["statistical_window"][
                "post"
            ][0],
            statistical_window_post_end=input_parameters["statistical_window"]["post"][
                1
            ],
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
            statistical_window_pre_start=input_parameters["statistical_window"]["pre"][
                0
            ],
            statistical_window_pre_end=input_parameters["statistical_window"]["pre"][1],
            statistical_window_post_start=input_parameters["statistical_window"][
                "post"
            ][0],
            statistical_window_post_end=input_parameters["statistical_window"]["post"][
                1
            ],
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
            statistical_window_pre_start=input_parameters["statistical_window"]["pre"][
                0
            ],
            statistical_window_pre_end=input_parameters["statistical_window"]["pre"][1],
            statistical_window_post_start=input_parameters["statistical_window"][
                "post"
            ][0],
            statistical_window_post_end=input_parameters["statistical_window"]["post"][
                1
            ],
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
            "output_metadata.json",
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
            os.path.join(self.data_dir, "cellset_series_part1-PCA-ICA.isxd"),
            os.path.join(self.data_dir, "cellset_series_part2-PCA-ICA.isxd"),
        ]
        compare_peri_event_activity_across_epochs(
            input_cellset_files=input_cellset_files,
            input_events_h5_file=self.input_events_h5_file,
            event_type=input_parameters["event_types"][0],
            visual_window_pre=input_parameters["visual_window"]["pre"],
            visual_window_post=input_parameters["visual_window"]["post"],
            statistical_window_pre_start=input_parameters["statistical_window"]["pre"][
                0
            ],
            statistical_window_pre_end=input_parameters["statistical_window"]["pre"][1],
            statistical_window_post_start=input_parameters["statistical_window"][
                "post"
            ][0],
            statistical_window_post_end=input_parameters["statistical_window"]["post"][
                1
            ],
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
            statistical_window_pre_start=input_parameters["statistical_window"]["pre"][
                0
            ],
            statistical_window_pre_end=input_parameters["statistical_window"]["pre"][1],
            statistical_window_post_start=input_parameters["statistical_window"][
                "post"
            ][0],
            statistical_window_post_end=input_parameters["statistical_window"]["post"][
                1
            ],
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
            statistical_window_pre_start=input_parameters["statistical_window"]["pre"][
                0
            ],
            statistical_window_pre_end=input_parameters["statistical_window"]["pre"][1],
            statistical_window_post_start=input_parameters["statistical_window"][
                "post"
            ][0],
            statistical_window_post_end=input_parameters["statistical_window"]["post"][
                1
            ],
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
            "output_metadata.json",
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
            os.path.join(self.data_dir, "cellset_series_part1-PCA-ICA.isxd"),
            os.path.join(self.data_dir, "cellset_series_part2-PCA-ICA.isxd"),
        ]
        compare_peri_event_activity_across_epochs(
            input_cellset_files=input_cellset_files,
            input_events_h5_file=self.input_events_h5_file,
            event_type=input_parameters["event_types"][0],
            visual_window_pre=input_parameters["visual_window"]["pre"],
            visual_window_post=input_parameters["visual_window"]["post"],
            statistical_window_pre_start=input_parameters["statistical_window"]["pre"][
                0
            ],
            statistical_window_pre_end=input_parameters["statistical_window"]["pre"][1],
            statistical_window_post_start=input_parameters["statistical_window"][
                "post"
            ][0],
            statistical_window_post_end=input_parameters["statistical_window"]["post"][
                1
            ],
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
            "output_metadata.json",
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
            os.path.join(self.data_dir, "cellset_series_part1-PCA-ICA.isxd"),
            os.path.join(self.data_dir, "cellset_series_part2-PCA-ICA.isxd"),
        ]
        compare_peri_event_activity_across_epochs(
            input_cellset_files=input_cellset_files,
            input_events_h5_file=self.input_events_h5_file,
            event_type=input_parameters["event_types"][0],
            visual_window_pre=input_parameters["visual_window"]["pre"],
            visual_window_post=input_parameters["visual_window"]["post"],
            statistical_window_pre_start=input_parameters["statistical_window"]["pre"][
                0
            ],
            statistical_window_pre_end=input_parameters["statistical_window"]["pre"][1],
            statistical_window_post_start=input_parameters["statistical_window"][
                "post"
            ][0],
            statistical_window_post_end=input_parameters["statistical_window"]["post"][
                1
            ],
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
            "output_metadata.json",
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
