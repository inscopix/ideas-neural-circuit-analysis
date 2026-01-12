import os
import shutil
import unittest

import numpy as np
import pandas as pd

from analysis.combine_compare_peri_event_data_across_epochs import (
    combine_compare_peri_event_data_across_epochs,
)


class TestCombineComparePeriEventDataAcrossEpochs(unittest.TestCase):
    """Tests for the combine and compare peri-event data across epochs tool."""

    # tool info
    tool_key = "combine_compare_peri_event_data_across_epochs"

    # define directories
    temporary_dir = "/tmp"
    input_dir = "data/combine_compare_peri_event_data_across_epochs"
    output_dir = os.path.join(
        temporary_dir, "tmp_combine_compare_peri_event_data_across_epochs"
    )

    # output manifest
    output_manifest_json_schema = "toolbox/tests/schemas/output_manifest_schema.json"
    output_manifest_file_basename = "output_manifest.json"
    output_manifest_file = os.path.join(output_dir, output_manifest_file_basename)

    # output metadata
    output_metadata_json_schema = "toolbox/tests/schemas/output_metadata_schema.json"
    output_metadata_file_basename = "output_metadata.json"
    output_metadata_file = os.path.join(output_dir, output_metadata_file_basename)

    def setUp(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def validate_traces_file_column_names(self, traces_df, epoch_names, cell_ids):
        """Validate output traces file column names."""
        exp_epoch_headers = epoch_names + ["Time"]
        act_epochs_headers = np.unique(traces_df.columns.get_level_values(0)).tolist()
        self.assertEqual(exp_epoch_headers, act_epochs_headers)
        self.assertEqual(traces_df.columns[0], ("Time", "Unnamed: 0_level_1"))

        for second_level_header in [
            "population_mean",
            "population_sem",
            "up_modulated_mean",
            "up_modulated_sem",
            "down_modulated_mean",
            "down_modulated_sem",
            "non_modulated_mean",
            "non_modulated_sem",
        ] + [f"C{str(i).zfill(2)}_mean" for i in cell_ids]:
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
            "file",
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

    def validate_group_anova_file_column_names(self, df):
        """Validate output ANOVA file column names."""
        exp_cols = [
            "Comparison",
            "Source",
            "SS",
            "DF1",
            "DF2",
            "MS",
            "F",
            "p-unc",
            "np2",
            "eps",
        ]
        act_cols = list(df.columns)
        self.assertEqual(exp_cols, act_cols)

    def validate_group_pairwise_file_column_names(self, df):
        """Validate output group pairwise comparison file column names."""
        exp_cols = [
            "Comparison",
            "Epoch",
            "Test",
            "Paired",
            "Parametric",
            "U-val",
            "alternative",
            "p-val",
            "RBC",
            "CLES",
        ]
        act_cols = list(df.columns)
        self.assertEqual(exp_cols, act_cols)

    def test_combine_single_group(self):
        """
        Validate that data from a single group can be combined (3 epochs).
        """
        # define input files
        group1_traces_files = [
            os.path.join(self.input_dir, f)
            for f in [
                "group1/group1_subject1.TRACES.csv",
                "group1/group1_subject2.TRACES.csv",
            ]
        ]
        group1_stats_files = [
            os.path.join(self.input_dir, f)
            for f in [
                "group1/group1_subject1.STATISTICS.csv",
                "group1/group1_subject2.STATISTICS.csv",
            ]
        ]

        # combine data
        group_name = "group1"
        epoch_names = ["Baseline", "Early Drug", "Late Drug"]
        combine_compare_peri_event_data_across_epochs(
            group1_traces_files=group1_traces_files,
            group1_stats_files=group1_stats_files,
            group1_name=group_name,
            group2_traces_files=[],
            group2_stats_files=[],
            group2_name="",
            epoch_names=", ".join(epoch_names),
            epoch_colors="tab:grey, tab:blue, tab:cyan",
            comparison_type="two_tailed",
            data_pairing="unpaired",
            parametric="auto",
            significance_threshold=0.05,
            average_method="neurons",
            tolerance=0.01,
            output_dir=self.output_dir,
        )

        # validate existence of expected output files
        actual_files = os.listdir(self.output_dir)
        expected_files = [
            # event-aligned traces
            f"event_aligned_activity_{group_name}.csv",
            f"event_aligned_population_activity_{group_name}.svg",
            f"event_aligned_population_activity_{group_name}_Baseline.preview.svg",
            f"event_aligned_population_activity_{group_name}_EarlyDrug.preview.svg",
            f"event_aligned_population_activity_{group_name}_LateDrug.preview.svg",
            f"event_aligned_activity_heatmap_{group_name}_Baseline.svg",
            f"event_aligned_activity_heatmap_{group_name}_EarlyDrug.svg",
            f"event_aligned_activity_heatmap_{group_name}_LateDrug.svg",
            # event-aligned statistics
            f"event_aligned_statistics_{group_name}.csv",
            f"event_aligned_activity_by_modulation_{group_name}_Baseline.preview.svg",
            f"event_aligned_activity_by_modulation_{group_name}_EarlyDrug.preview.svg",
            f"event_aligned_activity_by_modulation_{group_name}_LateDrug.preview.svg",
            f"fraction_of_modulated_neurons_{group_name}_Baseline.preview.svg",
            f"fraction_of_modulated_neurons_{group_name}_EarlyDrug.preview.svg",
            f"fraction_of_modulated_neurons_{group_name}_LateDrug.preview.svg",
            f"event_aligned_activity_up_modulated_{group_name}.svg",
            f"event_aligned_activity_down_modulated_{group_name}.svg",
            f"event_aligned_activity_non_modulated_{group_name}.svg",
            f"num_modulated_cells_per_epoch_{group_name}.svg",
            f"mean_post_minus_pre_activity_per_epoch_{group_name}.svg",
            # pairwise epoch comparisons
            f"pairwise_epoch_comparisons_{group_name}.csv",
            f"post_minus_pre_boxplot_{group_name}.svg",
            # manifest & metadata
            # "output_manifest.json",
            "output_metadata.json",
        ]
        self.assertEqual(sorted(actual_files), sorted(expected_files))

        # validate peri-event traces csv file
        traces_csv_file = os.path.join(
            self.output_dir,
            f"event_aligned_activity_{group_name}.csv",
        )
        traces_df = pd.read_csv(traces_csv_file, header=[0, 1])
        self.assertEqual(traces_df.shape, (41, 145))
        self.validate_traces_file_column_names(
            traces_df=traces_df, epoch_names=epoch_names, cell_ids=range(20)
        )

        # validate peri-event statistics file
        statistics_csv_file = os.path.join(
            self.output_dir, f"event_aligned_statistics_{group_name}.csv"
        )
        statistics_df = pd.read_csv(statistics_csv_file)
        self.assertEqual(statistics_df.shape, (120, 9))
        self.validate_statistics_file_column_names(statistics_df)

        # validate epoch comparison data file
        comparison_csv_file = os.path.join(
            self.output_dir, f"pairwise_epoch_comparisons_{group_name}.csv"
        )
        comparison_df = pd.read_csv(comparison_csv_file)
        self.assertEqual(comparison_df.shape, (3, 12))
        self.validate_pairwise_comparisons_file_column_names(comparison_df)

    def test_combine_single_group_average_by_recording(self):
        """
        Validate that data from a single group can be combined (3 epochs)
        when averaging the data by recording.
        """
        # define input files
        group1_traces_files = [
            os.path.join(self.input_dir, f)
            for f in [
                "group1/group1_subject1.TRACES.csv",
                "group1/group1_subject2.TRACES.csv",
            ]
        ]
        group1_stats_files = [
            os.path.join(self.input_dir, f)
            for f in [
                "group1/group1_subject1.STATISTICS.csv",
                "group1/group1_subject2.STATISTICS.csv",
            ]
        ]

        # combine data
        group_name = "group1"
        epoch_names = ["Baseline", "Early Drug", "Late Drug"]
        combine_compare_peri_event_data_across_epochs(
            group1_traces_files=group1_traces_files,
            group1_stats_files=group1_stats_files,
            group1_name=group_name,
            group2_traces_files=[],
            group2_stats_files=[],
            group2_name="",
            epoch_names=", ".join(epoch_names),
            epoch_colors="tab:grey, tab:blue, tab:cyan",
            comparison_type="two_tailed",
            data_pairing="unpaired",
            parametric="auto",
            significance_threshold=0.05,
            average_method="recordings",
            tolerance=0.01,
            output_dir=self.output_dir,
        )

        # validate existence of expected output files
        actual_files = os.listdir(self.output_dir)
        expected_files = [
            # event-aligned traces
            f"event_aligned_activity_{group_name}.csv",
            f"event_aligned_population_activity_{group_name}.svg",
            f"event_aligned_activity_by_modulation_{group_name}_Baseline.preview.svg",
            f"event_aligned_activity_by_modulation_{group_name}_EarlyDrug.preview.svg",
            f"event_aligned_activity_by_modulation_{group_name}_LateDrug.preview.svg",
            f"event_aligned_population_activity_{group_name}_Baseline.preview.svg",
            f"event_aligned_population_activity_{group_name}_EarlyDrug.preview.svg",
            f"event_aligned_population_activity_{group_name}_LateDrug.preview.svg",
            f"event_aligned_activity_heatmap_{group_name}_Baseline.svg",
            f"event_aligned_activity_heatmap_{group_name}_EarlyDrug.svg",
            f"event_aligned_activity_heatmap_{group_name}_LateDrug.svg",
            # event-aligned statistics
            f"event_aligned_statistics_{group_name}.csv",
            f"fraction_of_modulated_neurons_{group_name}_Baseline.preview.svg",
            f"fraction_of_modulated_neurons_{group_name}_EarlyDrug.preview.svg",
            f"fraction_of_modulated_neurons_{group_name}_LateDrug.preview.svg",
            f"event_aligned_activity_up_modulated_{group_name}.svg",
            f"event_aligned_activity_down_modulated_{group_name}.svg",
            f"event_aligned_activity_non_modulated_{group_name}.svg",
            f"num_modulated_cells_per_epoch_{group_name}.svg",
            f"mean_post_minus_pre_activity_per_epoch_{group_name}.svg",
            # pairwise epoch comparisons
            f"pairwise_epoch_comparisons_{group_name}.csv",
            f"post_minus_pre_boxplot_{group_name}.svg",
            # manifest & metadata
            # "output_manifest.json",
            "output_metadata.json",
        ]
        self.assertEqual(sorted(actual_files), sorted(expected_files))

        # validate peri-event traces csv file
        traces_csv_file = os.path.join(
            self.output_dir,
            f"event_aligned_activity_{group_name}.csv",
        )
        traces_df = pd.read_csv(traces_csv_file, header=[0, 1])
        self.assertEqual(traces_df.shape, (41, 145))
        self.validate_traces_file_column_names(
            traces_df=traces_df, epoch_names=epoch_names, cell_ids=range(20)
        )

        # validate peri-event statistics file
        statistics_csv_file = os.path.join(
            self.output_dir, f"event_aligned_statistics_{group_name}.csv"
        )
        statistics_df = pd.read_csv(statistics_csv_file)
        self.assertEqual(statistics_df.shape, (120, 9))
        self.validate_statistics_file_column_names(statistics_df)

        # validate epoch comparison data file
        comparison_csv_file = os.path.join(
            self.output_dir, f"pairwise_epoch_comparisons_{group_name}.csv"
        )
        comparison_df = pd.read_csv(comparison_csv_file)
        self.assertEqual(comparison_df.shape, (3, 12))
        self.validate_pairwise_comparisons_file_column_names(comparison_df)

    def test_combine_compare_two_groups(self):
        """
        Validate that data within each of the two input groups
        can be combined and compared across 3 epochs.
        """
        # define input files
        group1_traces_files = [
            os.path.join(self.input_dir, f)
            for f in [
                "group1/group1_subject1.TRACES.csv",
                "group1/group1_subject2.TRACES.csv",
            ]
        ]
        group1_stats_files = [
            os.path.join(self.input_dir, f)
            for f in [
                "group1/group1_subject1.STATISTICS.csv",
                "group1/group1_subject2.STATISTICS.csv",
            ]
        ]

        group2_traces_files = [
            os.path.join(self.input_dir, f)
            for f in [
                "group2/group2_subject1.TRACES.csv",
                "group2/group2_subject2.TRACES.csv",
            ]
        ]
        group2_stats_files = [
            os.path.join(self.input_dir, f)
            for f in [
                "group2/group2_subject1.STATISTICS.csv",
                "group2/group2_subject2.STATISTICS.csv",
            ]
        ]

        # combine data
        group1_name = "group1"
        group2_name = "group2"
        epoch_names = ["Baseline", "Early Drug", "Late Drug"]
        combine_compare_peri_event_data_across_epochs(
            group1_traces_files=group1_traces_files,
            group1_stats_files=group1_stats_files,
            group1_name=group1_name,
            group2_traces_files=group2_traces_files,
            group2_stats_files=group2_stats_files,
            group2_name=group2_name,
            epoch_names=", ".join(epoch_names),
            epoch_colors="tab:grey, tab:blue, tab:cyan",
            comparison_type="two_tailed",
            data_pairing="unpaired",
            parametric="auto",
            significance_threshold=0.05,
            average_method="neurons",
            tolerance=0.01,
            output_dir=self.output_dir,
        )

        # validate existence of expected output files
        actual_files = os.listdir(self.output_dir)
        expected_files = [
            # GROUP 1
            # event-aligned traces
            f"event_aligned_activity_{group1_name}.csv",
            f"event_aligned_population_activity_{group1_name}.svg",
            f"event_aligned_activity_by_modulation_{group1_name}_Baseline.preview.svg",
            f"event_aligned_activity_by_modulation_{group1_name}_EarlyDrug.preview.svg",
            f"event_aligned_activity_by_modulation_{group1_name}_LateDrug.preview.svg",
            f"event_aligned_population_activity_{group1_name}_Baseline.preview.svg",
            f"event_aligned_population_activity_{group1_name}_EarlyDrug.preview.svg",
            f"event_aligned_population_activity_{group1_name}_LateDrug.preview.svg",
            f"event_aligned_activity_heatmap_{group1_name}_Baseline.svg",
            f"event_aligned_activity_heatmap_{group1_name}_EarlyDrug.svg",
            f"event_aligned_activity_heatmap_{group1_name}_LateDrug.svg",
            # event-aligned statistics
            f"event_aligned_statistics_{group1_name}.csv",
            f"fraction_of_modulated_neurons_{group1_name}_Baseline.preview.svg",
            f"fraction_of_modulated_neurons_{group1_name}_EarlyDrug.preview.svg",
            f"fraction_of_modulated_neurons_{group1_name}_LateDrug.preview.svg",
            f"event_aligned_activity_up_modulated_{group1_name}.svg",
            f"event_aligned_activity_down_modulated_{group1_name}.svg",
            f"event_aligned_activity_non_modulated_{group1_name}.svg",
            f"num_modulated_cells_per_epoch_{group1_name}.svg",
            f"mean_post_minus_pre_activity_per_epoch_{group1_name}.svg",
            # pairwise epoch comparisons
            f"pairwise_epoch_comparisons_{group1_name}.csv",
            f"post_minus_pre_boxplot_{group1_name}.svg",
            # GROUP 2
            # event-aligned traces
            f"event_aligned_activity_{group2_name}.csv",
            f"event_aligned_population_activity_{group2_name}.svg",
            f"event_aligned_activity_by_modulation_{group2_name}_Baseline.preview.svg",
            f"event_aligned_activity_by_modulation_{group2_name}_EarlyDrug.preview.svg",
            f"event_aligned_activity_by_modulation_{group2_name}_LateDrug.preview.svg",
            f"event_aligned_population_activity_{group2_name}_Baseline.preview.svg",
            f"event_aligned_population_activity_{group2_name}_EarlyDrug.preview.svg",
            f"event_aligned_population_activity_{group2_name}_LateDrug.preview.svg",
            f"event_aligned_activity_heatmap_{group2_name}_Baseline.svg",
            f"event_aligned_activity_heatmap_{group2_name}_EarlyDrug.svg",
            f"event_aligned_activity_heatmap_{group2_name}_LateDrug.svg",
            # event-aligned statistics
            f"event_aligned_statistics_{group2_name}.csv",
            f"fraction_of_modulated_neurons_{group2_name}_Baseline.preview.svg",
            f"fraction_of_modulated_neurons_{group2_name}_EarlyDrug.preview.svg",
            f"fraction_of_modulated_neurons_{group2_name}_LateDrug.preview.svg",
            f"event_aligned_activity_up_modulated_{group2_name}.svg",
            f"event_aligned_activity_down_modulated_{group2_name}.svg",
            f"event_aligned_activity_non_modulated_{group2_name}.svg",
            f"num_modulated_cells_per_epoch_{group2_name}.svg",
            f"mean_post_minus_pre_activity_per_epoch_{group2_name}.svg",
            # pairwise epoch comparisons
            f"pairwise_epoch_comparisons_{group2_name}.csv",
            f"post_minus_pre_boxplot_{group2_name}.svg",
            # GROUP COMPARISON DATA
            "anova_group_comparisons.csv",
            "pairwise_group_comparisons.csv",
            "population_post_minus_pre_comparison.svg",
            # OTHER OUTPUT FILES
            # manifest & metadata
            # "output_manifest.json",
            "output_metadata.json",
        ]
        self.assertEqual(sorted(actual_files), sorted(expected_files))

        # Validate data for GROUP 1
        # validate peri-event traces csv file
        traces_csv_file = os.path.join(
            self.output_dir,
            f"event_aligned_activity_{group1_name}.csv",
        )
        traces_df = pd.read_csv(traces_csv_file, header=[0, 1])
        self.assertEqual(traces_df.shape, (41, 145))
        self.validate_traces_file_column_names(
            traces_df=traces_df, epoch_names=epoch_names, cell_ids=range(20)
        )

        # validate peri-event statistics file
        statistics_csv_file = os.path.join(
            self.output_dir, f"event_aligned_statistics_{group1_name}.csv"
        )
        statistics_df = pd.read_csv(statistics_csv_file)
        self.assertEqual(statistics_df.shape, (120, 9))
        self.validate_statistics_file_column_names(statistics_df)

        # validate epoch comparison data file
        comparison_csv_file = os.path.join(
            self.output_dir, f"pairwise_epoch_comparisons_{group1_name}.csv"
        )
        comparison_df = pd.read_csv(comparison_csv_file)
        self.assertEqual(comparison_df.shape, (3, 12))
        self.validate_pairwise_comparisons_file_column_names(comparison_df)

        # Validate data for GROUP 2
        # validate peri-event traces csv file
        traces_csv_file = os.path.join(
            self.output_dir,
            f"event_aligned_activity_{group2_name}.csv",
        )
        traces_df = pd.read_csv(traces_csv_file, header=[0, 1])
        self.assertEqual(traces_df.shape, (41, 145))
        self.validate_traces_file_column_names(
            traces_df=traces_df, epoch_names=epoch_names, cell_ids=range(20)
        )

        # validate peri-event statistics file
        statistics_csv_file = os.path.join(
            self.output_dir, f"event_aligned_statistics_{group2_name}.csv"
        )
        statistics_df = pd.read_csv(statistics_csv_file)
        self.assertEqual(statistics_df.shape, (120, 9))
        self.validate_statistics_file_column_names(statistics_df)

        # validate epoch comparison data file
        comparison_csv_file = os.path.join(
            self.output_dir, f"pairwise_epoch_comparisons_{group2_name}.csv"
        )
        comparison_df = pd.read_csv(comparison_csv_file)
        self.assertEqual(comparison_df.shape, (3, 12))

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
        act_cols = list(comparison_df.columns)
        self.assertEqual(exp_cols, act_cols)

        # Validate group comparison data
        group_anova_csv_file = os.path.join(
            self.output_dir, "anova_group_comparisons.csv"
        )
        group_anova_df = pd.read_csv(group_anova_csv_file)
        self.assertEqual(group_anova_df.shape, (3, 10))
        self.validate_group_anova_file_column_names(group_anova_df)

        group_pairwise_csv_file = os.path.join(
            self.output_dir, "pairwise_group_comparisons.csv"
        )
        group_pairwise_df = pd.read_csv(group_pairwise_csv_file)
        self.assertEqual(group_pairwise_df.shape, (3, 10))
        self.validate_group_pairwise_file_column_names(group_pairwise_df)

    def test_combine_data_when_group_name_includes_spaces(self):
        """
        Validate that group names with spaces are supported in the
        combine and compare peri-event data across epochs tool.
        """
        # define input files
        group1_traces_files = [
            os.path.join(self.input_dir, f)
            for f in [
                "group1/group1_subject1.TRACES.csv",
                "group1/group1_subject2.TRACES.csv",
            ]
        ]
        group1_stats_files = [
            os.path.join(self.input_dir, f)
            for f in [
                "group1/group1_subject1.STATISTICS.csv",
                "group1/group1_subject2.STATISTICS.csv",
            ]
        ]

        # combine data
        group_name = "My First Group"
        epoch_names = ["Baseline", "Early Drug", "Late Drug"]
        combine_compare_peri_event_data_across_epochs(
            group1_traces_files=group1_traces_files,
            group1_stats_files=group1_stats_files,
            group1_name=group_name,
            group2_traces_files=[],
            group2_stats_files=[],
            group2_name="",
            epoch_names=", ".join(epoch_names),
            epoch_colors="tab:grey, tab:blue, tab:cyan",
            comparison_type="two_tailed",
            data_pairing="unpaired",
            parametric="auto",
            significance_threshold=0.05,
            average_method="neurons",
            tolerance=0.01,
            output_dir=self.output_dir,
        )

        # validate existence of expected output files
        actual_files = os.listdir(self.output_dir)
        group_name = group_name.replace(" ", "")
        expected_files = [
            # event-aligned traces
            f"event_aligned_activity_{group_name}.csv",
            f"event_aligned_population_activity_{group_name}.svg",
            f"event_aligned_population_activity_{group_name}_Baseline.preview.svg",
            f"event_aligned_population_activity_{group_name}_EarlyDrug.preview.svg",
            f"event_aligned_population_activity_{group_name}_LateDrug.preview.svg",
            f"event_aligned_activity_heatmap_{group_name}_Baseline.svg",
            f"event_aligned_activity_heatmap_{group_name}_EarlyDrug.svg",
            f"event_aligned_activity_heatmap_{group_name}_LateDrug.svg",
            # event-aligned statistics
            f"event_aligned_statistics_{group_name}.csv",
            f"event_aligned_activity_by_modulation_{group_name}_Baseline.preview.svg",
            f"event_aligned_activity_by_modulation_{group_name}_EarlyDrug.preview.svg",
            f"event_aligned_activity_by_modulation_{group_name}_LateDrug.preview.svg",
            f"fraction_of_modulated_neurons_{group_name}_Baseline.preview.svg",
            f"fraction_of_modulated_neurons_{group_name}_EarlyDrug.preview.svg",
            f"fraction_of_modulated_neurons_{group_name}_LateDrug.preview.svg",
            f"event_aligned_activity_up_modulated_{group_name}.svg",
            f"event_aligned_activity_down_modulated_{group_name}.svg",
            f"event_aligned_activity_non_modulated_{group_name}.svg",
            f"num_modulated_cells_per_epoch_{group_name}.svg",
            f"mean_post_minus_pre_activity_per_epoch_{group_name}.svg",
            # pairwise epoch comparisons
            f"pairwise_epoch_comparisons_{group_name}.csv",
            f"post_minus_pre_boxplot_{group_name}.svg",
            # manifest & metadata
            # "output_manifest.json",
            "output_metadata.json",
        ]
        self.assertEqual(sorted(actual_files), sorted(expected_files))

        # validate peri-event traces csv file
        traces_csv_file = os.path.join(
            self.output_dir,
            f"event_aligned_activity_{group_name}.csv",
        )
        traces_df = pd.read_csv(traces_csv_file, header=[0, 1])
        self.assertEqual(traces_df.shape, (41, 145))
        self.validate_traces_file_column_names(
            traces_df=traces_df, epoch_names=epoch_names, cell_ids=range(20)
        )

        # validate peri-event statistics file
        statistics_csv_file = os.path.join(
            self.output_dir, f"event_aligned_statistics_{group_name}.csv"
        )
        statistics_df = pd.read_csv(statistics_csv_file)
        self.assertEqual(statistics_df.shape, (120, 9))
        self.validate_statistics_file_column_names(statistics_df)

        # validate epoch comparison data file
        comparison_csv_file = os.path.join(
            self.output_dir, f"pairwise_epoch_comparisons_{group_name}.csv"
        )
        comparison_df = pd.read_csv(comparison_csv_file)
        self.assertEqual(comparison_df.shape, (3, 12))
        self.validate_pairwise_comparisons_file_column_names(comparison_df)

    def test_combine_compare_single_epoch_baseline_data(self):
        """
        Validate that we can successfully combine and compare data
        from a single epoch across the two groups.
        """
        # define input files
        group1_traces_files = [
            os.path.join(self.input_dir, f)
            for f in [
                "single_epoch_baseline/group1_subject1.TRACES.csv",
                "single_epoch_baseline/group1_subject2.TRACES.csv",
            ]
        ]
        group1_stats_files = [
            os.path.join(self.input_dir, f)
            for f in [
                "single_epoch_baseline/group1_subject1.STATISTICS.csv",
                "single_epoch_baseline/group1_subject2.STATISTICS.csv",
            ]
        ]

        group2_traces_files = [
            os.path.join(self.input_dir, f)
            for f in [
                "single_epoch_baseline/group2_subject1.TRACES.csv",
                "single_epoch_baseline/group2_subject2.TRACES.csv",
            ]
        ]
        group2_stats_files = [
            os.path.join(self.input_dir, f)
            for f in [
                "single_epoch_baseline/group2_subject1.STATISTICS.csv",
                "single_epoch_baseline/group2_subject2.STATISTICS.csv",
            ]
        ]

        # combine data
        group1_name = "group1"
        group2_name = "group2"
        epoch_names = ["Baseline"]
        combine_compare_peri_event_data_across_epochs(
            group1_traces_files=group1_traces_files,
            group1_stats_files=group1_stats_files,
            group1_name=group1_name,
            group2_traces_files=group2_traces_files,
            group2_stats_files=group2_stats_files,
            group2_name=group2_name,
            epoch_names=", ".join(epoch_names),
            epoch_colors="blue",
            comparison_type="two_tailed",
            data_pairing="unpaired",
            parametric="auto",
            significance_threshold=0.05,
            average_method="neurons",
            tolerance=0.01,
            output_dir=self.output_dir,
        )

        # validate existence of expected output files
        actual_files = os.listdir(self.output_dir)
        expected_files = [
            # GROUP 1
            # event-aligned traces
            f"event_aligned_activity_{group1_name}.csv",
            f"event_aligned_population_activity_{group1_name}.svg",
            f"event_aligned_activity_by_modulation_{group1_name}_Baseline.preview.svg",
            f"event_aligned_population_activity_{group1_name}_Baseline.preview.svg",
            f"event_aligned_activity_heatmap_{group1_name}_Baseline.svg",
            # event-aligned statistics
            f"event_aligned_statistics_{group1_name}.csv",
            f"fraction_of_modulated_neurons_{group1_name}_Baseline.preview.svg",
            f"event_aligned_activity_up_modulated_{group1_name}.svg",
            f"event_aligned_activity_down_modulated_{group1_name}.svg",
            f"event_aligned_activity_non_modulated_{group1_name}.svg",
            f"num_modulated_cells_per_epoch_{group1_name}.svg",
            f"mean_post_minus_pre_activity_per_epoch_{group1_name}.svg",
            # pairwise epoch comparisons --> None generated since there is single epoch
            # GROUP 2
            # event-aligned traces
            f"event_aligned_activity_{group2_name}.csv",
            f"event_aligned_population_activity_{group2_name}.svg",
            f"event_aligned_activity_by_modulation_{group2_name}_Baseline.preview.svg",
            f"event_aligned_population_activity_{group2_name}_Baseline.preview.svg",
            f"event_aligned_activity_heatmap_{group2_name}_Baseline.svg",
            # event-aligned statistics
            f"event_aligned_statistics_{group2_name}.csv",
            f"fraction_of_modulated_neurons_{group2_name}_Baseline.preview.svg",
            f"event_aligned_activity_up_modulated_{group2_name}.svg",
            f"event_aligned_activity_down_modulated_{group2_name}.svg",
            f"event_aligned_activity_non_modulated_{group2_name}.svg",
            f"num_modulated_cells_per_epoch_{group2_name}.svg",
            f"mean_post_minus_pre_activity_per_epoch_{group2_name}.svg",
            # pairwise epoch comparisons --> None generated since there is single epoch
            # GROUP COMPARISON DATA
            # "anova_group_comparisons.csv", --> Not generated when there are less than 2 epochs
            "pairwise_group_comparisons.csv",
            "population_post_minus_pre_comparison.svg",
            # OTHER OUTPUT FILES
            # manifest & metadata
            # "output_manifest.json",
            "output_metadata.json",
        ]
        self.assertEqual(sorted(actual_files), sorted(expected_files))

        # Validate data for GROUP 1
        # validate peri-event traces csv file
        traces_csv_file = os.path.join(
            self.output_dir,
            f"event_aligned_activity_{group1_name}.csv",
        )
        traces_df = pd.read_csv(traces_csv_file, header=[0, 1])
        self.assertEqual(traces_df.shape, (41, 43))
        self.validate_traces_file_column_names(
            traces_df=traces_df, epoch_names=epoch_names, cell_ids=range(20)
        )

        # validate peri-event statistics file
        statistics_csv_file = os.path.join(
            self.output_dir, f"event_aligned_statistics_{group1_name}.csv"
        )
        statistics_df = pd.read_csv(statistics_csv_file)
        self.assertEqual(statistics_df.shape, (34, 9))
        self.validate_statistics_file_column_names(statistics_df)

        # validate epoch comparison data file
        comparison_csv_file = os.path.join(
            self.output_dir, f"pairwise_epoch_comparisons_{group1_name}.csv"
        )
        self.assertFalse(os.path.exists(comparison_csv_file))

        # Validate data for GROUP 2
        # validate peri-event traces csv file
        traces_csv_file = os.path.join(
            self.output_dir,
            f"event_aligned_activity_{group2_name}.csv",
        )
        traces_df = pd.read_csv(traces_csv_file, header=[0, 1])
        self.assertEqual(traces_df.shape, (41, 25))

        # validate peri-event statistics file
        statistics_csv_file = os.path.join(
            self.output_dir, f"event_aligned_statistics_{group2_name}.csv"
        )
        statistics_df = pd.read_csv(statistics_csv_file)
        self.assertEqual(statistics_df.shape, (16, 9))
        self.validate_statistics_file_column_names(statistics_df)

        # validate epoch comparison data file
        comparison_csv_file = os.path.join(
            self.output_dir, f"pairwise_epoch_comparisons_{group2_name}.csv"
        )
        self.assertFalse(os.path.exists(comparison_csv_file))

        # Validate group comparison data
        group_anova_csv_file = os.path.join(
            self.output_dir, "anova_group_comparisons.csv"
        )
        self.assertFalse(os.path.exists(group_anova_csv_file))

        group_pairwise_csv_file = os.path.join(
            self.output_dir, "pairwise_group_comparisons.csv"
        )
        group_pairwise_df = pd.read_csv(group_pairwise_csv_file)
        self.assertEqual(group_pairwise_df.shape, (1, 13))

        exp_cols = [
            "Comparison",
            "Epoch",
            "Test",
            "Paired",
            "Parametric",
            "T",
            "dof",
            "alternative",
            "p-val",
            "CI95%",
            "cohen-d",
            "BF10",
            "power",
        ]
        act_cols = list(group_pairwise_df.columns)
        self.assertEqual(exp_cols, act_cols)

    def test_combine_compare_two_groups_with_different_epochs(self):
        """
        Validate that data from two groups containing different epoch names
        can be compared.
        - Data within each group should be combined as usual.
        - Comparisons should only be performed for epochs for which boths groups contain data.
        """
        # define input files
        group1_traces_files = [
            os.path.join(self.input_dir, f)
            for f in [
                "group1/group1_subject1.TRACES.csv",
                "group1/group1_subject2.TRACES.csv",
            ]
        ]
        group1_stats_files = [
            os.path.join(self.input_dir, f)
            for f in [
                "group1/group1_subject1.STATISTICS.csv",
                "group1/group1_subject2.STATISTICS.csv",
            ]
        ]

        group2_traces_files = [
            os.path.join(self.input_dir, f)
            for f in [
                "single_epoch_baseline/group2_subject1.TRACES.csv",
                "single_epoch_baseline/group2_subject2.TRACES.csv",
            ]
        ]
        group2_stats_files = [
            os.path.join(self.input_dir, f)
            for f in [
                "single_epoch_baseline/group2_subject1.STATISTICS.csv",
                "single_epoch_baseline/group2_subject2.STATISTICS.csv",
            ]
        ]

        # combine data
        group1_name = "group1"
        group2_name = "group2"
        epoch_names = ["Baseline", "Early Drug", "Late Drug"]
        combine_compare_peri_event_data_across_epochs(
            group1_traces_files=group1_traces_files,
            group1_stats_files=group1_stats_files,
            group1_name=group1_name,
            group2_traces_files=group2_traces_files,
            group2_stats_files=group2_stats_files,
            group2_name=group2_name,
            epoch_names=", ".join(epoch_names),
            epoch_colors="tab:grey, tab:blue, tab:cyan",
            comparison_type="two_tailed",
            data_pairing="unpaired",
            parametric="False",
            significance_threshold=0.05,
            average_method="neurons",
            tolerance=0.01,
            output_dir=self.output_dir,
        )

        # validate existence of expected output files
        actual_files = os.listdir(self.output_dir)
        expected_files = [
            # GROUP 1
            # event-aligned traces
            f"event_aligned_activity_{group1_name}.csv",
            f"event_aligned_population_activity_{group1_name}.svg",
            f"event_aligned_activity_by_modulation_{group1_name}_Baseline.preview.svg",
            f"event_aligned_activity_by_modulation_{group1_name}_EarlyDrug.preview.svg",
            f"event_aligned_activity_by_modulation_{group1_name}_LateDrug.preview.svg",
            f"event_aligned_population_activity_{group1_name}_Baseline.preview.svg",
            f"event_aligned_population_activity_{group1_name}_EarlyDrug.preview.svg",
            f"event_aligned_population_activity_{group1_name}_LateDrug.preview.svg",
            f"event_aligned_activity_heatmap_{group1_name}_Baseline.svg",
            f"event_aligned_activity_heatmap_{group1_name}_EarlyDrug.svg",
            f"event_aligned_activity_heatmap_{group1_name}_LateDrug.svg",
            # event-aligned statistics
            f"event_aligned_statistics_{group1_name}.csv",
            f"fraction_of_modulated_neurons_{group1_name}_Baseline.preview.svg",
            f"fraction_of_modulated_neurons_{group1_name}_EarlyDrug.preview.svg",
            f"fraction_of_modulated_neurons_{group1_name}_LateDrug.preview.svg",
            f"event_aligned_activity_up_modulated_{group1_name}.svg",
            f"event_aligned_activity_down_modulated_{group1_name}.svg",
            f"event_aligned_activity_non_modulated_{group1_name}.svg",
            f"num_modulated_cells_per_epoch_{group1_name}.svg",
            f"mean_post_minus_pre_activity_per_epoch_{group1_name}.svg",
            # pairwise epoch comparisons
            f"pairwise_epoch_comparisons_{group1_name}.csv",
            f"post_minus_pre_boxplot_{group1_name}.svg",
            # GROUP 2
            # event-aligned traces
            f"event_aligned_activity_{group2_name}.csv",
            f"event_aligned_population_activity_{group2_name}.svg",
            f"event_aligned_activity_by_modulation_{group2_name}_Baseline.preview.svg",
            f"event_aligned_population_activity_{group2_name}_Baseline.preview.svg",
            f"event_aligned_activity_heatmap_{group2_name}_Baseline.svg",
            # event-aligned statistics
            f"event_aligned_statistics_{group2_name}.csv",
            f"fraction_of_modulated_neurons_{group2_name}_Baseline.preview.svg",
            f"event_aligned_activity_up_modulated_{group2_name}.svg",
            f"event_aligned_activity_down_modulated_{group2_name}.svg",
            f"event_aligned_activity_non_modulated_{group2_name}.svg",
            f"num_modulated_cells_per_epoch_{group2_name}.svg",
            f"mean_post_minus_pre_activity_per_epoch_{group2_name}.svg",
            # pairwise epoch comparisons --> single epoch so no epoch comparisons
            # GROUP COMPARISON DATA
            "pairwise_group_comparisons.csv",
            "population_post_minus_pre_comparison.svg",
            # OTHER OUTPUT FILES
            # manifest & metadata
            # "output_manifest.json",
            "output_metadata.json",
        ]
        self.assertEqual(sorted(actual_files), sorted(expected_files))

        # Validate data for GROUP 1
        # validate peri-event traces csv file
        traces_csv_file = os.path.join(
            self.output_dir,
            f"event_aligned_activity_{group1_name}.csv",
        )
        traces_df = pd.read_csv(traces_csv_file, header=[0, 1])
        self.assertEqual(traces_df.shape, (41, 145))
        self.validate_traces_file_column_names(
            traces_df=traces_df, epoch_names=epoch_names, cell_ids=range(20)
        )

        # validate peri-event statistics file
        statistics_csv_file = os.path.join(
            self.output_dir, f"event_aligned_statistics_{group1_name}.csv"
        )
        statistics_df = pd.read_csv(statistics_csv_file)
        self.assertEqual(statistics_df.shape, (120, 9))
        self.validate_statistics_file_column_names(statistics_df)
        self.assertEqual(statistics_df["epoch"].unique().tolist(), epoch_names)

        # validate epoch comparison data file
        comparison_csv_file = os.path.join(
            self.output_dir, f"pairwise_epoch_comparisons_{group1_name}.csv"
        )
        comparison_df = pd.read_csv(comparison_csv_file)
        self.assertEqual(comparison_df.shape, (3, 12))
        self.validate_pairwise_comparisons_file_column_names(comparison_df)

        # Validate data for GROUP 2
        # validate peri-event traces csv file
        traces_csv_file = os.path.join(
            self.output_dir,
            f"event_aligned_activity_{group2_name}.csv",
        )
        traces_df = pd.read_csv(traces_csv_file, header=[0, 1])
        self.assertEqual(traces_df.shape, (41, 25))
        self.validate_traces_file_column_names(
            traces_df=traces_df,
            epoch_names=["Baseline"],
            cell_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19],
        )

        # validate peri-event statistics file
        statistics_csv_file = os.path.join(
            self.output_dir, f"event_aligned_statistics_{group2_name}.csv"
        )
        statistics_df = pd.read_csv(statistics_csv_file)
        self.assertEqual(statistics_df.shape, (16, 9))
        self.validate_statistics_file_column_names(statistics_df)

        # validate epoch comparison data file
        comparison_csv_file = os.path.join(
            self.output_dir, f"pairwise_epoch_comparisons_{group2_name}.csv"
        )
        self.assertFalse(os.path.exists(comparison_csv_file))

        # Validate group comparison data
        group_anova_csv_file = os.path.join(
            self.output_dir, "anova_group_comparisons.csv"
        )
        self.assertFalse(os.path.exists(group_anova_csv_file))

        group_pairwise_csv_file = os.path.join(
            self.output_dir, "pairwise_group_comparisons.csv"
        )
        group_pairwise_df = pd.read_csv(group_pairwise_csv_file)
        self.assertEqual(group_pairwise_df.shape, (1, 10))
        self.validate_group_pairwise_file_column_names(group_pairwise_df)

    def test_combine_compare_groups_with_non_overlapping_epochs(self):
        """
        Validate that we can successfully combine data within each group
        and still succeed without comparing the groups when there are no
        epochs in common between the two groups.
        """
        # define input files
        group1_traces_files = [
            os.path.join(self.input_dir, f)
            for f in [
                "single_epoch_drug/group1_subject1.TRACES.csv",
                "single_epoch_drug/group1_subject2.TRACES.csv",
            ]
        ]
        group1_stats_files = [
            os.path.join(self.input_dir, f)
            for f in [
                "single_epoch_drug/group1_subject1.STATISTICS.csv",
                "single_epoch_drug/group1_subject2.STATISTICS.csv",
            ]
        ]

        group2_traces_files = [
            os.path.join(self.input_dir, f)
            for f in [
                "single_epoch_baseline/group2_subject1.TRACES.csv",
                "single_epoch_baseline/group2_subject2.TRACES.csv",
            ]
        ]
        group2_stats_files = [
            os.path.join(self.input_dir, f)
            for f in [
                "single_epoch_baseline/group2_subject1.STATISTICS.csv",
                "single_epoch_baseline/group2_subject2.STATISTICS.csv",
            ]
        ]

        # combine data
        group1_name = "DrugGroup"
        group2_name = "BaselineGroup"
        epoch_names = ["Baseline", "Drug"]
        combine_compare_peri_event_data_across_epochs(
            group1_traces_files=group1_traces_files,
            group1_stats_files=group1_stats_files,
            group1_name=group1_name,
            group2_traces_files=group2_traces_files,
            group2_stats_files=group2_stats_files,
            group2_name=group2_name,
            epoch_names=", ".join(epoch_names),
            epoch_colors="grey, blue",
            comparison_type="two_tailed",
            data_pairing="unpaired",
            parametric="auto",
            significance_threshold=0.05,
            average_method="neurons",
            tolerance=0.01,
            output_dir=self.output_dir,
        )

        # validate existence of expected output files
        actual_files = os.listdir(self.output_dir)
        expected_files = [
            # GROUP 1
            # event-aligned traces
            f"event_aligned_activity_{group1_name}.csv",
            f"event_aligned_population_activity_{group1_name}.svg",
            f"event_aligned_activity_by_modulation_{group1_name}_Drug.preview.svg",
            f"event_aligned_population_activity_{group1_name}_Drug.preview.svg",
            f"event_aligned_activity_heatmap_{group1_name}_Drug.svg",
            # event-aligned statistics
            f"event_aligned_statistics_{group1_name}.csv",
            f"fraction_of_modulated_neurons_{group1_name}_Drug.preview.svg",
            f"event_aligned_activity_up_modulated_{group1_name}.svg",
            f"event_aligned_activity_down_modulated_{group1_name}.svg",
            f"event_aligned_activity_non_modulated_{group1_name}.svg",
            f"num_modulated_cells_per_epoch_{group1_name}.svg",
            f"mean_post_minus_pre_activity_per_epoch_{group1_name}.svg",
            # pairwise epoch comparisons --> None generated since there is single epoch
            # GROUP 2
            # event-aligned traces
            f"event_aligned_activity_{group2_name}.csv",
            f"event_aligned_population_activity_{group2_name}.svg",
            f"event_aligned_activity_by_modulation_{group2_name}_Baseline.preview.svg",
            f"event_aligned_population_activity_{group2_name}_Baseline.preview.svg",
            f"event_aligned_activity_heatmap_{group2_name}_Baseline.svg",
            # event-aligned statistics
            f"event_aligned_statistics_{group2_name}.csv",
            f"fraction_of_modulated_neurons_{group2_name}_Baseline.preview.svg",
            f"event_aligned_activity_up_modulated_{group2_name}.svg",
            f"event_aligned_activity_down_modulated_{group2_name}.svg",
            f"event_aligned_activity_non_modulated_{group2_name}.svg",
            f"num_modulated_cells_per_epoch_{group2_name}.svg",
            f"mean_post_minus_pre_activity_per_epoch_{group2_name}.svg",
            # pairwise epoch comparisons --> None generated since there is single epoch
            # GROUP COMPARISON DATA --> not generated since there are no epoch in common between the groups
            # OTHER OUTPUT FILES
            # manifest & metadata
            # "output_manifest.json",
            "output_metadata.json",
        ]
        self.assertEqual(sorted(actual_files), sorted(expected_files))

        # Validate data for GROUP 1
        # validate peri-event traces csv file
        traces_csv_file = os.path.join(
            self.output_dir,
            f"event_aligned_activity_{group1_name}.csv",
        )
        traces_df = pd.read_csv(traces_csv_file, header=[0, 1])
        self.assertEqual(traces_df.shape, (41, 43))
        self.validate_traces_file_column_names(
            traces_df=traces_df, epoch_names=["Drug"], cell_ids=range(20)
        )

        # validate peri-event statistics file
        statistics_csv_file = os.path.join(
            self.output_dir, f"event_aligned_statistics_{group1_name}.csv"
        )
        statistics_df = pd.read_csv(statistics_csv_file)
        self.assertEqual(statistics_df.shape, (34, 9))
        self.validate_statistics_file_column_names(statistics_df)

        # validate epoch comparison data file
        comparison_csv_file = os.path.join(
            self.output_dir, f"pairwise_epoch_comparisons_{group1_name}.csv"
        )
        self.assertFalse(os.path.exists(comparison_csv_file))

        # Validate data for GROUP 2
        # validate peri-event traces csv file
        traces_csv_file = os.path.join(
            self.output_dir,
            f"event_aligned_activity_{group2_name}.csv",
        )
        traces_df = pd.read_csv(traces_csv_file, header=[0, 1])
        self.assertEqual(traces_df.shape, (41, 25))

        # validate peri-event statistics file
        statistics_csv_file = os.path.join(
            self.output_dir, f"event_aligned_statistics_{group2_name}.csv"
        )
        statistics_df = pd.read_csv(statistics_csv_file)
        self.assertEqual(statistics_df.shape, (16, 9))
        self.validate_statistics_file_column_names(statistics_df)

        # validate epoch comparison data file
        comparison_csv_file = os.path.join(
            self.output_dir, f"pairwise_epoch_comparisons_{group2_name}.csv"
        )
        self.assertFalse(os.path.exists(comparison_csv_file))

        # Validate group comparison data
        group_anova_csv_file = os.path.join(
            self.output_dir, "anova_group_comparisons.csv"
        )
        self.assertFalse(os.path.exists(group_anova_csv_file))

        group_pairwise_csv_file = os.path.join(
            self.output_dir, "pairwise_group_comparisons.csv"
        )
        self.assertFalse(os.path.exists(group_pairwise_csv_file))

    def test_combine_single_group_with_non_overlapping_epochs_across_subjects(
        self,
    ):
        """
        Validate that we can successfully combine data within a given group
        even if the epochs specified for each subject do not overlap.
        """
        # define input files
        group1_traces_files = [
            os.path.join(self.input_dir, f)
            for f in [
                "single_epoch_baseline/group1_subject1.TRACES.csv",
                "single_epoch_baseline/group1_subject2.TRACES.csv",
                "single_epoch_drug/group1_subject1.TRACES.csv",
                "single_epoch_drug/group1_subject2.TRACES.csv",
            ]
        ]
        group1_stats_files = [
            os.path.join(self.input_dir, f)
            for f in [
                "single_epoch_baseline/group1_subject1.STATISTICS.csv",
                "single_epoch_baseline/group1_subject2.STATISTICS.csv",
                "single_epoch_drug/group1_subject1.STATISTICS.csv",
                "single_epoch_drug/group1_subject2.STATISTICS.csv",
            ]
        ]

        # combine data
        epoch_names = ["Baseline", "Drug"]
        group_name = "group1"
        combine_compare_peri_event_data_across_epochs(
            group1_traces_files=group1_traces_files,
            group1_stats_files=group1_stats_files,
            group1_name=group_name,
            group2_traces_files=[],
            group2_stats_files=[],
            group2_name="",
            epoch_names=", ".join(epoch_names),
            epoch_colors="tab:grey, tab:blue",
            comparison_type="two_tailed",
            data_pairing="unpaired",
            parametric="auto",
            significance_threshold=0.05,
            average_method="neurons",
            tolerance=0.01,
            output_dir=self.output_dir,
        )

        # validate existence of expected output files
        actual_files = os.listdir(self.output_dir)
        group_name = group_name.replace(" ", "")
        expected_files = [
            # event-aligned traces
            f"event_aligned_activity_{group_name}.csv",
            f"event_aligned_population_activity_{group_name}.svg",
            f"event_aligned_population_activity_{group_name}_Baseline.preview.svg",
            f"event_aligned_population_activity_{group_name}_Drug.preview.svg",
            f"event_aligned_activity_heatmap_{group_name}_Baseline.svg",
            f"event_aligned_activity_heatmap_{group_name}_Drug.svg",
            # event-aligned statistics
            f"event_aligned_statistics_{group_name}.csv",
            f"event_aligned_activity_by_modulation_{group_name}_Baseline.preview.svg",
            f"event_aligned_activity_by_modulation_{group_name}_Drug.preview.svg",
            f"fraction_of_modulated_neurons_{group_name}_Baseline.preview.svg",
            f"fraction_of_modulated_neurons_{group_name}_Drug.preview.svg",
            f"event_aligned_activity_up_modulated_{group_name}.svg",
            f"event_aligned_activity_down_modulated_{group_name}.svg",
            f"event_aligned_activity_non_modulated_{group_name}.svg",
            f"num_modulated_cells_per_epoch_{group_name}.svg",
            f"mean_post_minus_pre_activity_per_epoch_{group_name}.svg",
            # pairwise epoch comparisons
            f"pairwise_epoch_comparisons_{group_name}.csv",
            f"post_minus_pre_boxplot_{group_name}.svg",
            # manifest & metadata
            # "output_manifest.json",
            "output_metadata.json",
        ]
        self.assertEqual(sorted(actual_files), sorted(expected_files))

        # validate peri-event traces csv file
        traces_csv_file = os.path.join(
            self.output_dir,
            f"event_aligned_activity_{group_name}.csv",
        )
        traces_df = pd.read_csv(traces_csv_file, header=[0, 1])
        self.assertEqual(traces_df.shape, (41, 85))
        self.validate_traces_file_column_names(
            traces_df=traces_df, epoch_names=epoch_names, cell_ids=range(20)
        )

        # validate peri-event statistics file
        statistics_csv_file = os.path.join(
            self.output_dir, f"event_aligned_statistics_{group_name}.csv"
        )
        statistics_df = pd.read_csv(statistics_csv_file)
        self.assertEqual(statistics_df.shape, (68, 9))
        self.validate_statistics_file_column_names(statistics_df)

        # validate epoch comparison data file
        comparison_csv_file = os.path.join(
            self.output_dir, f"pairwise_epoch_comparisons_{group_name}.csv"
        )
        comparison_df = pd.read_csv(comparison_csv_file)
        self.assertEqual(comparison_df.shape, (1, 12))
        self.validate_pairwise_comparisons_file_column_names(comparison_df)
