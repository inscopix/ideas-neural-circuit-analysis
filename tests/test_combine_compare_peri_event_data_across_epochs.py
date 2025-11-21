import os
import shutil
import unittest
import pandas as pd
import numpy as np
from analysis.combine_compare_peri_event_data_across_epochs import (
    combine_compare_peri_event_data_across_epochs,
)

class TestCombineComparePeriEventDataAcrossEpochs(unittest.TestCase):
    """Tests for the combine and compare peri-event data across epochs tool."""

    # tool info
    tool_key = "combine_compare_peri_event_data_across_epochs"

    # define directories
    temporary_dir = "/tmp"
    input_dir = (
        "data/combine_compare_peri_event_data_across_epochs"
    )
    output_dir = os.path.join(
        temporary_dir, "tmp_combine_compare_peri_event_data_across_epochs"
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

    def setUp(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

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
            "file",
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
        print("ACT COLS ===> ", comparison_df.columns)
        self.validate_pairwise_comparisons_file_column_names(comparison_df)

        # validate output manifest
        # exp_manifest = {
        #     "schema_version": "2.0.0",
        #     "groups": [
        #         {
        #             "group_key": "combine_compare_peri_event_data_across_epochs_output",
        #             "group_type": "tool_output",
        #             "group_id": "cff4a357-3a3c-40f0-875f-0616d6544c73",
        #             "series": [],
        #             "files": [
        #                 {
        #                     "file_key": "group1_traces_files",
        #                     "file_name": "group1_subject1.TRACES.csv",
        #                     "file_id": "b6d87004-b08b-4dca-9e58-8b419206c3dd",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/group1/group1_subject1.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_traces_files",
        #                     "file_name": "group1_subject2.TRACES.csv",
        #                     "file_id": "41cfc4ee-e3ce-4bdf-8871-296639ac5911",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/group1/group1_subject2.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_stats_files",
        #                     "file_name": "group1_subject1.STATISTICS.csv",
        #                     "file_id": "db8ac686-2a47-49bc-9ad0-117d7089a282",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/group1/group1_subject1.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_stats_files",
        #                     "file_name": "group1_subject2.STATISTICS.csv",
        #                     "file_id": "bb567711-fb33-4833-ba0b-197afd2cca1e",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/group1/group1_subject2.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_event_aligned_traces",
        #                     "file_name": "event_aligned_activity_group1.csv",
        #                     "file_id": "54fad201-55bd-481b-9c00-70b5082a13e0",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_group1.csv",
        #                     "file_type": "combined_peri_event_traces",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned population activity across epochs",
        #                             "help": "Comparison of event-aligned average population activity across the epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned population activity",
        #                             "help": "Event-aligned average population activity line plot (epoch: Baseline).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_group1_Baseline.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned population activity",
        #                             "help": "Event-aligned average population activity line plot (epoch: Early Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_group1_EarlyDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned population activity",
        #                             "help": "Event-aligned average population activity line plot (epoch: Late Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_group1_LateDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap (epoch: Baseline)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_heatmap_group1_Baseline.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap (epoch: Early Drug)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_heatmap_group1_EarlyDrug.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap (epoch: Late Drug)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_heatmap_group1_LateDrug.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group1_event_aligned_statistics",
        #                     "file_name": "event_aligned_statistics_group1.csv",
        #                     "file_id": "d6bd9884-7cde-44b1-8d37-8461d8a46e1a",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_statistics_group1.csv",
        #                     "file_type": "combined_peri_event_statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Baseline).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_by_modulation_group1_Baseline.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Early Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_by_modulation_group1_EarlyDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Late Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_by_modulation_group1_LateDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of up-modulated cells",
        #                             "help": "Comparison of event-aligned activity of up-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_up_modulated_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of down-modulated cells",
        #                             "help": "Comparison of event-aligned activity of down-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_down_modulated_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of non-modulated cells",
        #                             "help": "Comparison of event-aligned activity of non-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_non_modulated_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in each sub-population (up-, down-, and non-modulated neurons) (epoch: Baseline).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/fraction_of_modulated_neurons_group1_Baseline.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in each sub-population (up-, down-, and non-modulated neurons) (epoch: Early Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/fraction_of_modulated_neurons_group1_EarlyDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in each sub-population (up-, down-, and non-modulated neurons) (epoch: Late Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/fraction_of_modulated_neurons_group1_LateDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Number of modulated cells per epoch",
        #                             "help": "Number of up-, down-, and non-modulated neurons per epoch.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/num_modulated_cells_per_epoch_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Mean post-pre activity per epoch",
        #                             "help": "Comparison of mean post-pre activity across the epochs. The error bars represent the standard error of the mean.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/mean_post_minus_pre_activity_per_epoch_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group1_epoch_comparison_data",
        #                     "file_name": "pairwise_epoch_comparisons_group1.csv",
        #                     "file_id": "3867b6cb-c357-46d5-b391-4fcdb09af676",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/pairwise_epoch_comparisons_group1.csv",
        #                     "file_type": "peri_event_comparison_data",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "parent_ids": [
        #                         "54fad201-55bd-481b-9c00-70b5082a13e0",
        #                         "d6bd9884-7cde-44b1-8d37-8461d8a46e1a",
        #                     ],
        #                     "preview": [
        #                         {
        #                             "name": "Post-pre distribution across the epochs",
        #                             "help": "Distribution of post-pre activity across epochs displayed using a box plot. Lines connect the same cells together.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/post_minus_pre_boxplot_group1.svg",
        #                             "file_format": "svg",
        #                         }
        #                     ],
        #                 },
        #             ],
        #         }
        #     ],
        # }
        # self.assertTrue(
        #     validate_output_manifest(
        #         expected_output_manifest=exp_manifest,
        #         tool_metadata=self.tool_metadata[self.tool_key],
        #         actual_manifest_data_filepath=self.output_manifest_file,
        #     )
        # )

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

        # validate output manifest
        # exp_manifest = {
        #     "schema_version": "2.0.0",
        #     "groups": [
        #         {
        #             "group_key": "combine_compare_peri_event_data_across_epochs_output",
        #             "group_type": "tool_output",
        #             "group_id": "cff4a357-3a3c-40f0-875f-0616d6544c73",
        #             "series": [],
        #             "files": [
        #                 {
        #                     "file_key": "group1_traces_files",
        #                     "file_name": "group1_subject1.TRACES.csv",
        #                     "file_id": "b6d87004-b08b-4dca-9e58-8b419206c3dd",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/group1/group1_subject1.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_traces_files",
        #                     "file_name": "group1_subject2.TRACES.csv",
        #                     "file_id": "41cfc4ee-e3ce-4bdf-8871-296639ac5911",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/group1/group1_subject2.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_stats_files",
        #                     "file_name": "group1_subject1.STATISTICS.csv",
        #                     "file_id": "db8ac686-2a47-49bc-9ad0-117d7089a282",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/group1/group1_subject1.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_stats_files",
        #                     "file_name": "group1_subject2.STATISTICS.csv",
        #                     "file_id": "bb567711-fb33-4833-ba0b-197afd2cca1e",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/group1/group1_subject2.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_event_aligned_traces",
        #                     "file_name": "event_aligned_activity_group1.csv",
        #                     "file_id": "54fad201-55bd-481b-9c00-70b5082a13e0",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_group1.csv",
        #                     "file_type": "combined_peri_event_traces",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned population activity across epochs",
        #                             "help": "Comparison of event-aligned average population activity across the epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned population activity",
        #                             "help": "Event-aligned average population activity line plot (epoch: Baseline).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_group1_Baseline.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned population activity",
        #                             "help": "Event-aligned average population activity line plot (epoch: Early Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_group1_EarlyDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned population activity",
        #                             "help": "Event-aligned average population activity line plot (epoch: Late Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_group1_LateDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap (epoch: Baseline)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_heatmap_group1_Baseline.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap (epoch: Early Drug)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_heatmap_group1_EarlyDrug.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap (epoch: Late Drug)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_heatmap_group1_LateDrug.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group1_event_aligned_statistics",
        #                     "file_name": "event_aligned_statistics_group1.csv",
        #                     "file_id": "d6bd9884-7cde-44b1-8d37-8461d8a46e1a",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_statistics_group1.csv",
        #                     "file_type": "combined_peri_event_statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Baseline).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_by_modulation_group1_Baseline.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Early Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_by_modulation_group1_EarlyDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Late Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_by_modulation_group1_LateDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of up-modulated cells",
        #                             "help": "Comparison of event-aligned activity of up-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_up_modulated_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of down-modulated cells",
        #                             "help": "Comparison of event-aligned activity of down-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_down_modulated_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of non-modulated cells",
        #                             "help": "Comparison of event-aligned activity of non-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_non_modulated_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in each sub-population (up-, down-, and non-modulated neurons) (epoch: Baseline).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/fraction_of_modulated_neurons_group1_Baseline.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in each sub-population (up-, down-, and non-modulated neurons) (epoch: Early Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/fraction_of_modulated_neurons_group1_EarlyDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in each sub-population (up-, down-, and non-modulated neurons) (epoch: Late Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/fraction_of_modulated_neurons_group1_LateDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Number of modulated cells per epoch",
        #                             "help": "Number of up-, down-, and non-modulated neurons per epoch.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/num_modulated_cells_per_epoch_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Mean post-pre activity per epoch",
        #                             "help": "Comparison of mean post-pre activity across the epochs. The error bars represent the standard error of the mean.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/mean_post_minus_pre_activity_per_epoch_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group1_epoch_comparison_data",
        #                     "file_name": "pairwise_epoch_comparisons_group1.csv",
        #                     "file_id": "3867b6cb-c357-46d5-b391-4fcdb09af676",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/pairwise_epoch_comparisons_group1.csv",
        #                     "file_type": "peri_event_comparison_data",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "parent_ids": [
        #                         "54fad201-55bd-481b-9c00-70b5082a13e0",
        #                         "d6bd9884-7cde-44b1-8d37-8461d8a46e1a",
        #                     ],
        #                     "preview": [
        #                         {
        #                             "name": "Post-pre distribution across the epochs",
        #                             "help": "Distribution of post-pre activity across epochs displayed using a box plot. Lines connect the same cells together.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/post_minus_pre_boxplot_group1.svg",
        #                             "file_format": "svg",
        #                         }
        #                     ],
        #                 },
        #             ],
        #         }
        #     ],
        # }
        # self.assertTrue(
        #     validate_output_manifest(
        #         expected_output_manifest=exp_manifest,
        #         tool_metadata=self.tool_metadata[self.tool_key],
        #         actual_manifest_data_filepath=self.output_manifest_file,
        #     )
        # )

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

        # # validate output manifest
        # exp_manifest = {
        #     "schema_version": "2.0.0",
        #     "groups": [
        #         {
        #             "group_key": "combine_compare_peri_event_data_across_epochs_output",
        #             "group_type": "tool_output",
        #             "group_id": "5a434009-6e0d-48f5-ad8d-1487a69cfbe6",
        #             "series": [],
        #             "files": [
        #                 {
        #                     "file_key": "group1_traces_files",
        #                     "file_name": "group1_subject1.TRACES.csv",
        #                     "file_id": "d38ebde8-df9b-40a3-ac98-fe6838d7f2e4",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/group1/group1_subject1.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_traces_files",
        #                     "file_name": "group1_subject2.TRACES.csv",
        #                     "file_id": "ce18e11a-a1f8-47a2-91a1-0c5f64e570ef",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/group1/group1_subject2.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_stats_files",
        #                     "file_name": "group1_subject1.STATISTICS.csv",
        #                     "file_id": "9a4b4dbc-d073-40b6-b5f2-23cbad1f989c",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/group1/group1_subject1.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_stats_files",
        #                     "file_name": "group1_subject2.STATISTICS.csv",
        #                     "file_id": "647ac29d-054e-4e75-b3d5-6e63ac4c9237",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/group1/group1_subject2.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group2_traces_files",
        #                     "file_name": "group2_subject1.TRACES.csv",
        #                     "file_id": "35630ec1-f1de-4de4-af0d-e9d914c030d0",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/group2/group2_subject1.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group2_traces_files",
        #                     "file_name": "group2_subject2.TRACES.csv",
        #                     "file_id": "ab73ac5f-4b56-4835-8238-6b845b608bdd",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/group2/group2_subject2.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group2_stats_files",
        #                     "file_name": "group2_subject1.STATISTICS.csv",
        #                     "file_id": "5b12aed5-7944-4a2a-a0fa-28796d42de67",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/group2/group2_subject1.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group2_stats_files",
        #                     "file_name": "group2_subject2.STATISTICS.csv",
        #                     "file_id": "9f6cef4f-b17b-419e-921d-dba7db139b30",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/group2/group2_subject2.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_event_aligned_traces",
        #                     "file_name": "event_aligned_activity_group1.csv",
        #                     "file_id": "6693e1a2-4daf-460b-8c04-1ab59ea7bd59",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_group1.csv",
        #                     "file_type": "combined_peri_event_traces",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned population activity across epochs",
        #                             "help": "Comparison of event-aligned average population activity across the epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned population activity",
        #                             "help": "Event-aligned average population activity line plot (epoch: Baseline).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_group1_Baseline.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned population activity",
        #                             "help": "Event-aligned average population activity line plot (epoch: Early Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_group1_EarlyDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned population activity",
        #                             "help": "Event-aligned average population activity line plot (epoch: Late Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_group1_LateDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap (epoch: Baseline)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_heatmap_group1_Baseline.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap (epoch: Early Drug)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_heatmap_group1_EarlyDrug.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap (epoch: Late Drug)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_heatmap_group1_LateDrug.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group1_event_aligned_statistics",
        #                     "file_name": "event_aligned_statistics_group1.csv",
        #                     "file_id": "51fb6aa2-420b-4652-acd8-c48f18c77857",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_statistics_group1.csv",
        #                     "file_type": "combined_peri_event_statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Baseline).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_by_modulation_group1_Baseline.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Early Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_by_modulation_group1_EarlyDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Late Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_by_modulation_group1_LateDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of up-modulated cells",
        #                             "help": "Comparison of event-aligned activity of up-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_up_modulated_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of down-modulated cells",
        #                             "help": "Comparison of event-aligned activity of down-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_down_modulated_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of non-modulated cells",
        #                             "help": "Comparison of event-aligned activity of non-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_non_modulated_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in each sub-population (up-, down-, and non-modulated neurons) (epoch: Baseline).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/fraction_of_modulated_neurons_group1_Baseline.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in each sub-population (up-, down-, and non-modulated neurons) (epoch: Early Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/fraction_of_modulated_neurons_group1_EarlyDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in each sub-population (up-, down-, and non-modulated neurons) (epoch: Late Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/fraction_of_modulated_neurons_group1_LateDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Number of modulated cells per epoch",
        #                             "help": "Number of up-, down-, and non-modulated neurons per epoch.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/num_modulated_cells_per_epoch_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Mean post-pre activity per epoch",
        #                             "help": "Comparison of mean post-pre activity across the epochs. The error bars represent the standard error of the mean.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/mean_post_minus_pre_activity_per_epoch_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group1_epoch_comparison_data",
        #                     "file_name": "pairwise_epoch_comparisons_group1.csv",
        #                     "file_id": "87fc9d46-30f1-41c9-83c9-cf99b3a23814",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/pairwise_epoch_comparisons_group1.csv",
        #                     "file_type": "peri_event_comparison_data",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "parent_ids": [
        #                         "6693e1a2-4daf-460b-8c04-1ab59ea7bd59",
        #                         "51fb6aa2-420b-4652-acd8-c48f18c77857",
        #                     ],
        #                     "preview": [
        #                         {
        #                             "name": "Post-pre distribution across the epochs",
        #                             "help": "Distribution of post-pre activity across epochs displayed using a box plot. Lines connect the same cells together.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/post_minus_pre_boxplot_group1.svg",
        #                             "file_format": "svg",
        #                         }
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group2_event_aligned_traces",
        #                     "file_name": "event_aligned_activity_group2.csv",
        #                     "file_id": "93d29aaa-6ad1-49b3-9254-0693fe62b117",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_group2.csv",
        #                     "file_type": "combined_peri_event_traces",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned population activity across epochs",
        #                             "help": "Comparison of event-aligned average population activity across the epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_group2.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned population activity",
        #                             "help": "Event-aligned average population activity line plot (epoch: Baseline).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_group2_Baseline.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned population activity",
        #                             "help": "Event-aligned average population activity line plot (epoch: Early Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_group2_EarlyDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned population activity",
        #                             "help": "Event-aligned average population activity line plot (epoch: Late Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_group2_LateDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap (epoch: Baseline)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_heatmap_group2_Baseline.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap (epoch: Early Drug)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_heatmap_group2_EarlyDrug.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap (epoch: Late Drug)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_heatmap_group2_LateDrug.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group2_event_aligned_statistics",
        #                     "file_name": "event_aligned_statistics_group2.csv",
        #                     "file_id": "3cb682ae-8776-4bbe-ad98-e555298edd9f",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_statistics_group2.csv",
        #                     "file_type": "combined_peri_event_statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Baseline).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_by_modulation_group2_Baseline.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Early Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_by_modulation_group2_EarlyDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Late Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_by_modulation_group2_LateDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of up-modulated cells",
        #                             "help": "Comparison of event-aligned activity of up-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_up_modulated_group2.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of down-modulated cells",
        #                             "help": "Comparison of event-aligned activity of down-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_down_modulated_group2.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of non-modulated cells",
        #                             "help": "Comparison of event-aligned activity of non-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_non_modulated_group2.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in each sub-population (up-, down-, and non-modulated neurons) (epoch: Baseline).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/fraction_of_modulated_neurons_group2_Baseline.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in each sub-population (up-, down-, and non-modulated neurons) (epoch: Early Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/fraction_of_modulated_neurons_group2_EarlyDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in each sub-population (up-, down-, and non-modulated neurons) (epoch: Late Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/fraction_of_modulated_neurons_group2_LateDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Number of modulated cells per epoch",
        #                             "help": "Number of up-, down-, and non-modulated neurons per epoch.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/num_modulated_cells_per_epoch_group2.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Mean post-pre activity per epoch",
        #                             "help": "Comparison of mean post-pre activity across the epochs. The error bars represent the standard error of the mean.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/mean_post_minus_pre_activity_per_epoch_group2.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group2_epoch_comparison_data",
        #                     "file_name": "pairwise_epoch_comparisons_group2.csv",
        #                     "file_id": "188d39f5-8315-4a79-9bd4-94547b361eb4",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/pairwise_epoch_comparisons_group2.csv",
        #                     "file_type": "peri_event_comparison_data",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "parent_ids": [
        #                         "93d29aaa-6ad1-49b3-9254-0693fe62b117",
        #                         "3cb682ae-8776-4bbe-ad98-e555298edd9f",
        #                     ],
        #                     "preview": [
        #                         {
        #                             "name": "Post-pre distribution across the epochs",
        #                             "help": "Distribution of post-pre activity across epochs displayed using a box plot. Lines connect the same cells together.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/post_minus_pre_boxplot_group2.svg",
        #                             "file_format": "svg",
        #                         }
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group_anova_comparisons",
        #                     "file_name": "anova_group_comparisons.csv",
        #                     "file_id": "8a578a79-a1b9-4b70-9b00-ff1435c33a90",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/anova_group_comparisons.csv",
        #                     "file_type": "peri_event_comparison_data",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Comparisons of post-pre activity between the two groups",
        #                             "help": "Comparisons of post-pre activity between the two groups",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/population_post_minus_pre_comparison.svg",
        #                             "file_format": "svg",
        #                         }
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group_pairwise_comparisons",
        #                     "file_name": "pairwise_group_comparisons.csv",
        #                     "file_id": "e01d617f-2b4f-45b5-9fef-0ef8bfe03412",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/pairwise_group_comparisons.csv",
        #                     "file_type": "peri_event_comparison_data",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Comparisons of post-pre activity between the two groups",
        #                             "help": "Comparisons of post-pre activity between the two groups",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/population_post_minus_pre_comparison.svg",
        #                             "file_format": "svg",
        #                         }
        #                     ],
        #                 },
        #             ],
        #         }
        #     ],
        # }
        # self.assertTrue(
        #     validate_output_manifest(
        #         expected_output_manifest=exp_manifest,
        #         tool_metadata=self.tool_metadata[self.tool_key],
        #         actual_manifest_data_filepath=self.output_manifest_file,
        #     )
        # )

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

        # # validate output manifest
        # exp_manifest = {
        #     "schema_version": "2.0.0",
        #     "groups": [
        #         {
        #             "group_key": "combine_compare_peri_event_data_across_epochs_output",
        #             "group_type": "tool_output",
        #             "group_id": "dd857425-f6e4-408b-b625-2da61ebbe984",
        #             "series": [],
        #             "files": [
        #                 {
        #                     "file_key": "group1_traces_files",
        #                     "file_name": "group1_subject1.TRACES.csv",
        #                     "file_id": "ee58725b-7d6d-4e9e-93fb-31803fd13cc8",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/group1/group1_subject1.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_traces_files",
        #                     "file_name": "group1_subject2.TRACES.csv",
        #                     "file_id": "a4978173-afe7-44b4-9551-a7ef74851922",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/group1/group1_subject2.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_stats_files",
        #                     "file_name": "group1_subject1.STATISTICS.csv",
        #                     "file_id": "3082bc6a-e006-4562-a6ac-b43676ca4ba8",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/group1/group1_subject1.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_stats_files",
        #                     "file_name": "group1_subject2.STATISTICS.csv",
        #                     "file_id": "45a75749-124e-4c60-a122-89643927c88a",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/group1/group1_subject2.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_event_aligned_traces",
        #                     "file_name": "event_aligned_activity_MyFirstGroup.csv",
        #                     "file_id": "2899d271-7db4-48a0-bdea-8f29b5586c9a",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_MyFirstGroup.csv",
        #                     "file_type": "combined_peri_event_traces",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned population activity across epochs",
        #                             "help": "Comparison of event-aligned average population activity across the epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_MyFirstGroup.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned population activity",
        #                             "help": "Event-aligned average population activity line plot (epoch: Baseline).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_MyFirstGroup_Baseline.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned population activity",
        #                             "help": "Event-aligned average population activity line plot (epoch: Early Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_MyFirstGroup_EarlyDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned population activity",
        #                             "help": "Event-aligned average population activity line plot (epoch: Late Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_MyFirstGroup_LateDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap (epoch: Baseline)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_heatmap_MyFirstGroup_Baseline.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap (epoch: Early Drug)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_heatmap_MyFirstGroup_EarlyDrug.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap (epoch: Late Drug)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_heatmap_MyFirstGroup_LateDrug.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group1_event_aligned_statistics",
        #                     "file_name": "event_aligned_statistics_MyFirstGroup.csv",
        #                     "file_id": "00b30cd4-a254-4ef7-b5a4-28e2f3b13fb7",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_statistics_MyFirstGroup.csv",
        #                     "file_type": "combined_peri_event_statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Baseline).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_by_modulation_MyFirstGroup_Baseline.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Early Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_by_modulation_MyFirstGroup_EarlyDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Late Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_by_modulation_MyFirstGroup_LateDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of up-modulated cells",
        #                             "help": "Comparison of event-aligned activity of up-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_up_modulated_MyFirstGroup.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of down-modulated cells",
        #                             "help": "Comparison of event-aligned activity of down-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_down_modulated_MyFirstGroup.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of non-modulated cells",
        #                             "help": "Comparison of event-aligned activity of non-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_non_modulated_MyFirstGroup.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in each sub-population (up-, down-, and non-modulated neurons) (epoch: Baseline).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/fraction_of_modulated_neurons_MyFirstGroup_Baseline.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in each sub-population (up-, down-, and non-modulated neurons) (epoch: Early Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/fraction_of_modulated_neurons_MyFirstGroup_EarlyDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in each sub-population (up-, down-, and non-modulated neurons) (epoch: Late Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/fraction_of_modulated_neurons_MyFirstGroup_LateDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Number of modulated cells per epoch",
        #                             "help": "Number of up-, down-, and non-modulated neurons per epoch.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/num_modulated_cells_per_epoch_MyFirstGroup.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Mean post-pre activity per epoch",
        #                             "help": "Comparison of mean post-pre activity across the epochs. The error bars represent the standard error of the mean.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/mean_post_minus_pre_activity_per_epoch_MyFirstGroup.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group1_epoch_comparison_data",
        #                     "file_name": "pairwise_epoch_comparisons_MyFirstGroup.csv",
        #                     "file_id": "97bcc613-ac4d-4af6-b5de-47e6bfb79203",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/pairwise_epoch_comparisons_MyFirstGroup.csv",
        #                     "file_type": "peri_event_comparison_data",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "parent_ids": [
        #                         "2899d271-7db4-48a0-bdea-8f29b5586c9a",
        #                         "00b30cd4-a254-4ef7-b5a4-28e2f3b13fb7",
        #                     ],
        #                     "preview": [
        #                         {
        #                             "name": "Post-pre distribution across the epochs",
        #                             "help": "Distribution of post-pre activity across epochs displayed using a box plot. Lines connect the same cells together.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/post_minus_pre_boxplot_MyFirstGroup.svg",
        #                             "file_format": "svg",
        #                         }
        #                     ],
        #                 },
        #             ],
        #         }
        #     ],
        # }
        # self.assertTrue(
        #     validate_output_manifest(
        #         expected_output_manifest=exp_manifest,
        #         tool_metadata=self.tool_metadata[self.tool_key],
        #         actual_manifest_data_filepath=self.output_manifest_file,
        #     )
        # )

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

        # # validate output manifest
        # exp_manifest = {
        #     "schema_version": "2.0.0",
        #     "groups": [
        #         {
        #             "group_key": "combine_compare_peri_event_data_across_epochs_output",
        #             "group_type": "tool_output",
        #             "group_id": "e1cada87-def0-449c-8482-fefe12af473c",
        #             "series": [],
        #             "files": [
        #                 {
        #                     "file_key": "group1_traces_files",
        #                     "file_name": "group1_subject1.TRACES.csv",
        #                     "file_id": "3592abc7-237c-45c1-a1d7-e8cf8ec83b78",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/single_epoch_baseline/group1_subject1.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_traces_files",
        #                     "file_name": "group1_subject2.TRACES.csv",
        #                     "file_id": "014ae867-b5f6-4b48-a4d5-19ee9fe0a07b",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/single_epoch_baseline/group1_subject2.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_stats_files",
        #                     "file_name": "group1_subject1.STATISTICS.csv",
        #                     "file_id": "f6200889-e963-49e8-a500-3b4a67d012fe",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/single_epoch_baseline/group1_subject1.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_stats_files",
        #                     "file_name": "group1_subject2.STATISTICS.csv",
        #                     "file_id": "04e8c5f4-31d0-46cd-a9b1-8965ca32cefa",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/single_epoch_baseline/group1_subject2.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group2_traces_files",
        #                     "file_name": "group2_subject1.TRACES.csv",
        #                     "file_id": "962bbd30-0d2e-4107-87c7-330c3e290521",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/single_epoch_baseline/group2_subject1.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group2_traces_files",
        #                     "file_name": "group2_subject2.TRACES.csv",
        #                     "file_id": "36307630-3b1b-41dc-ac67-9da2d60f1dac",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/single_epoch_baseline/group2_subject2.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group2_stats_files",
        #                     "file_name": "group2_subject1.STATISTICS.csv",
        #                     "file_id": "59f6dca0-8117-4523-9145-010d9097b8e8",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/single_epoch_baseline/group2_subject1.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group2_stats_files",
        #                     "file_name": "group2_subject2.STATISTICS.csv",
        #                     "file_id": "706e8508-2683-41ca-99c3-17a353ebb9b5",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/single_epoch_baseline/group2_subject2.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_event_aligned_traces",
        #                     "file_name": "event_aligned_activity_group1.csv",
        #                     "file_id": "79a4a509-7ae3-4313-87d6-4f5b32832d16",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_group1.csv",
        #                     "file_type": "combined_peri_event_traces",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned population activity across epochs",
        #                             "help": "Comparison of event-aligned average population activity across the epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned population activity",
        #                             "help": "Event-aligned average population activity line plot (epoch: Baseline).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_group1_Baseline.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap (epoch: Baseline)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_heatmap_group1_Baseline.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group1_event_aligned_statistics",
        #                     "file_name": "event_aligned_statistics_group1.csv",
        #                     "file_id": "3123a39e-8827-46d6-9856-8f959ae95fe0",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_statistics_group1.csv",
        #                     "file_type": "combined_peri_event_statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Baseline).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_by_modulation_group1_Baseline.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of up-modulated cells",
        #                             "help": "Comparison of event-aligned activity of up-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_up_modulated_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of down-modulated cells",
        #                             "help": "Comparison of event-aligned activity of down-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_down_modulated_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of non-modulated cells",
        #                             "help": "Comparison of event-aligned activity of non-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_non_modulated_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in each sub-population (up-, down-, and non-modulated neurons) (epoch: Baseline).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/fraction_of_modulated_neurons_group1_Baseline.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Number of modulated cells per epoch",
        #                             "help": "Number of up-, down-, and non-modulated neurons per epoch.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/num_modulated_cells_per_epoch_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Mean post-pre activity per epoch",
        #                             "help": "Comparison of mean post-pre activity across the epochs. The error bars represent the standard error of the mean.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/mean_post_minus_pre_activity_per_epoch_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group2_event_aligned_traces",
        #                     "file_name": "event_aligned_activity_group2.csv",
        #                     "file_id": "57ee20cb-89cc-4c26-9bfb-2a962b7003ed",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_group2.csv",
        #                     "file_type": "combined_peri_event_traces",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned population activity across epochs",
        #                             "help": "Comparison of event-aligned average population activity across the epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_group2.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned population activity",
        #                             "help": "Event-aligned average population activity line plot (epoch: Baseline).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_group2_Baseline.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap (epoch: Baseline)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_heatmap_group2_Baseline.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group2_event_aligned_statistics",
        #                     "file_name": "event_aligned_statistics_group2.csv",
        #                     "file_id": "47a8139f-0e78-41ad-92d2-5e2ea42519d9",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_statistics_group2.csv",
        #                     "file_type": "combined_peri_event_statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Baseline).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_by_modulation_group2_Baseline.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of up-modulated cells",
        #                             "help": "Comparison of event-aligned activity of up-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_up_modulated_group2.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of down-modulated cells",
        #                             "help": "Comparison of event-aligned activity of down-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_down_modulated_group2.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of non-modulated cells",
        #                             "help": "Comparison of event-aligned activity of non-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_non_modulated_group2.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in each sub-population (up-, down-, and non-modulated neurons) (epoch: Baseline).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/fraction_of_modulated_neurons_group2_Baseline.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Number of modulated cells per epoch",
        #                             "help": "Number of up-, down-, and non-modulated neurons per epoch.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/num_modulated_cells_per_epoch_group2.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Mean post-pre activity per epoch",
        #                             "help": "Comparison of mean post-pre activity across the epochs. The error bars represent the standard error of the mean.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/mean_post_minus_pre_activity_per_epoch_group2.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group_pairwise_comparisons",
        #                     "file_name": "pairwise_group_comparisons.csv",
        #                     "file_id": "c4dadae0-e80f-46f4-b436-7fa2eee816fc",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/pairwise_group_comparisons.csv",
        #                     "file_type": "peri_event_comparison_data",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Comparisons of post-pre activity between the two groups",
        #                             "help": "Comparisons of post-pre activity between the two groups",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/population_post_minus_pre_comparison.svg",
        #                             "file_format": "svg",
        #                         }
        #                     ],
        #                 },
        #             ],
        #         }
        #     ],
        # }
        # self.assertTrue(
        #     validate_output_manifest(
        #         expected_output_manifest=exp_manifest,
        #         tool_metadata=self.tool_metadata[self.tool_key],
        #         actual_manifest_data_filepath=self.output_manifest_file,
        #     )
        # )

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

        # # validate output manifest
        # exp_manifest = {
        #     "schema_version": "2.0.0",
        #     "groups": [
        #         {
        #             "group_key": "combine_compare_peri_event_data_across_epochs_output",
        #             "group_type": "tool_output",
        #             "group_id": "06536524-468e-47a6-9f43-e7e58adf58f1",
        #             "series": [],
        #             "files": [
        #                 {
        #                     "file_key": "group1_traces_files",
        #                     "file_name": "group1_subject1.TRACES.csv",
        #                     "file_id": "8446c189-7d94-4086-8800-8b9c6cd4ba2e",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/group1/group1_subject1.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_traces_files",
        #                     "file_name": "group1_subject2.TRACES.csv",
        #                     "file_id": "83c7a5a5-4b6a-43d7-a9d1-832bfa4b31c2",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/group1/group1_subject2.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_stats_files",
        #                     "file_name": "group1_subject1.STATISTICS.csv",
        #                     "file_id": "a77be7e4-b45d-46fb-98c4-73a3e621794e",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/group1/group1_subject1.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_stats_files",
        #                     "file_name": "group1_subject2.STATISTICS.csv",
        #                     "file_id": "1a78fd97-1ebb-45f1-ad77-7eaeb9e81362",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/group1/group1_subject2.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group2_traces_files",
        #                     "file_name": "group2_subject1.TRACES.csv",
        #                     "file_id": "dc98508c-0d82-4d57-abd3-7de2bf81d44c",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/single_epoch_baseline/group2_subject1.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group2_traces_files",
        #                     "file_name": "group2_subject2.TRACES.csv",
        #                     "file_id": "4bd08ba6-128f-44df-9b7e-cd14649ae8f1",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/single_epoch_baseline/group2_subject2.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group2_stats_files",
        #                     "file_name": "group2_subject1.STATISTICS.csv",
        #                     "file_id": "6a4f9ba0-e91f-46f1-aa5b-fdf227b9ffdb",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/single_epoch_baseline/group2_subject1.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group2_stats_files",
        #                     "file_name": "group2_subject2.STATISTICS.csv",
        #                     "file_id": "8ff86dc5-2039-4c03-bf29-09ff11d5e925",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/single_epoch_baseline/group2_subject2.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_event_aligned_traces",
        #                     "file_name": "event_aligned_activity_group1.csv",
        #                     "file_id": "6971396e-efbf-48fe-9e0c-78c9d431990a",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_group1.csv",
        #                     "file_type": "combined_peri_event_traces",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned population activity across epochs",
        #                             "help": "Comparison of event-aligned average population activity across the epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned population activity",
        #                             "help": "Event-aligned average population activity line plot (epoch: Baseline).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_group1_Baseline.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned population activity",
        #                             "help": "Event-aligned average population activity line plot (epoch: Early Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_group1_EarlyDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned population activity",
        #                             "help": "Event-aligned average population activity line plot (epoch: Late Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_group1_LateDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap (epoch: Baseline)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_heatmap_group1_Baseline.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap (epoch: Early Drug)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_heatmap_group1_EarlyDrug.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap (epoch: Late Drug)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_heatmap_group1_LateDrug.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group1_event_aligned_statistics",
        #                     "file_name": "event_aligned_statistics_group1.csv",
        #                     "file_id": "bf7697e4-55f3-4cf5-9756-7d63c0e0c160",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_statistics_group1.csv",
        #                     "file_type": "combined_peri_event_statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Baseline).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_by_modulation_group1_Baseline.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Early Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_by_modulation_group1_EarlyDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Late Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_by_modulation_group1_LateDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of up-modulated cells",
        #                             "help": "Comparison of event-aligned activity of up-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_up_modulated_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of down-modulated cells",
        #                             "help": "Comparison of event-aligned activity of down-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_down_modulated_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of non-modulated cells",
        #                             "help": "Comparison of event-aligned activity of non-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_non_modulated_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in each sub-population (up-, down-, and non-modulated neurons) (epoch: Baseline).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/fraction_of_modulated_neurons_group1_Baseline.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in each sub-population (up-, down-, and non-modulated neurons) (epoch: Early Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/fraction_of_modulated_neurons_group1_EarlyDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in each sub-population (up-, down-, and non-modulated neurons) (epoch: Late Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/fraction_of_modulated_neurons_group1_LateDrug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Number of modulated cells per epoch",
        #                             "help": "Number of up-, down-, and non-modulated neurons per epoch.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/num_modulated_cells_per_epoch_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Mean post-pre activity per epoch",
        #                             "help": "Comparison of mean post-pre activity across the epochs. The error bars represent the standard error of the mean.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/mean_post_minus_pre_activity_per_epoch_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group1_epoch_comparison_data",
        #                     "file_name": "pairwise_epoch_comparisons_group1.csv",
        #                     "file_id": "134991cc-2298-4cab-903e-03d921e03b56",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/pairwise_epoch_comparisons_group1.csv",
        #                     "file_type": "peri_event_comparison_data",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "parent_ids": [
        #                         "6971396e-efbf-48fe-9e0c-78c9d431990a",
        #                         "bf7697e4-55f3-4cf5-9756-7d63c0e0c160",
        #                     ],
        #                     "preview": [
        #                         {
        #                             "name": "Post-pre distribution across the epochs",
        #                             "help": "Distribution of post-pre activity across epochs displayed using a box plot. Lines connect the same cells together.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/post_minus_pre_boxplot_group1.svg",
        #                             "file_format": "svg",
        #                         }
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group2_event_aligned_traces",
        #                     "file_name": "event_aligned_activity_group2.csv",
        #                     "file_id": "b10aaf90-f930-42eb-aa7a-c77af5a90b80",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_group2.csv",
        #                     "file_type": "combined_peri_event_traces",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned population activity across epochs",
        #                             "help": "Comparison of event-aligned average population activity across the epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_group2.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned population activity",
        #                             "help": "Event-aligned average population activity line plot (epoch: Baseline).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_group2_Baseline.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap (epoch: Baseline)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_heatmap_group2_Baseline.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group2_event_aligned_statistics",
        #                     "file_name": "event_aligned_statistics_group2.csv",
        #                     "file_id": "bc10dae1-3848-44da-8f90-6d24b2a858a4",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_statistics_group2.csv",
        #                     "file_type": "combined_peri_event_statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Baseline).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_by_modulation_group2_Baseline.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of up-modulated cells",
        #                             "help": "Comparison of event-aligned activity of up-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_up_modulated_group2.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of down-modulated cells",
        #                             "help": "Comparison of event-aligned activity of down-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_down_modulated_group2.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of non-modulated cells",
        #                             "help": "Comparison of event-aligned activity of non-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_non_modulated_group2.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in each sub-population (up-, down-, and non-modulated neurons) (epoch: Baseline).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/fraction_of_modulated_neurons_group2_Baseline.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Number of modulated cells per epoch",
        #                             "help": "Number of up-, down-, and non-modulated neurons per epoch.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/num_modulated_cells_per_epoch_group2.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Mean post-pre activity per epoch",
        #                             "help": "Comparison of mean post-pre activity across the epochs. The error bars represent the standard error of the mean.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/mean_post_minus_pre_activity_per_epoch_group2.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group_pairwise_comparisons",
        #                     "file_name": "pairwise_group_comparisons.csv",
        #                     "file_id": "a7341ff2-8dab-4861-ae7f-f8690ab5cf37",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/pairwise_group_comparisons.csv",
        #                     "file_type": "peri_event_comparison_data",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Comparisons of post-pre activity between the two groups",
        #                             "help": "Comparisons of post-pre activity between the two groups",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/population_post_minus_pre_comparison.svg",
        #                             "file_format": "svg",
        #                         }
        #                     ],
        #                 },
        #             ],
        #         }
        #     ],
        # }
        # self.assertTrue(
        #     validate_output_manifest(
        #         expected_output_manifest=exp_manifest,
        #         tool_metadata=self.tool_metadata[self.tool_key],
        #         actual_manifest_data_filepath=self.output_manifest_file,
        #     )
        # )

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

        # # validate output manifest
        # exp_manifest = {
        #     "schema_version": "2.0.0",
        #     "groups": [
        #         {
        #             "group_key": "combine_compare_peri_event_data_across_epochs_output",
        #             "group_type": "tool_output",
        #             "group_id": "8fc8081c-b351-47ae-9697-9798de5b0a45",
        #             "series": [],
        #             "files": [
        #                 {
        #                     "file_key": "group1_traces_files",
        #                     "file_name": "group1_subject1.TRACES.csv",
        #                     "file_id": "3a52429c-0650-4dfa-a5fb-44a5b798da14",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/single_epoch_drug/group1_subject1.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_traces_files",
        #                     "file_name": "group1_subject2.TRACES.csv",
        #                     "file_id": "d77e2fcd-5e55-4047-8305-0757f488ee25",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/single_epoch_drug/group1_subject2.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_stats_files",
        #                     "file_name": "group1_subject1.STATISTICS.csv",
        #                     "file_id": "026de122-5709-41bd-a0f8-71e326da301f",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/single_epoch_drug/group1_subject1.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_stats_files",
        #                     "file_name": "group1_subject2.STATISTICS.csv",
        #                     "file_id": "c1955e03-315d-4726-a423-bb808dbc19c9",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/single_epoch_drug/group1_subject2.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group2_traces_files",
        #                     "file_name": "group2_subject1.TRACES.csv",
        #                     "file_id": "90e408f6-6653-493e-81de-b18dae26f3f8",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/single_epoch_baseline/group2_subject1.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group2_traces_files",
        #                     "file_name": "group2_subject2.TRACES.csv",
        #                     "file_id": "ca81c10c-2673-4b21-b1d9-a85bd491268f",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/single_epoch_baseline/group2_subject2.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group2_stats_files",
        #                     "file_name": "group2_subject1.STATISTICS.csv",
        #                     "file_id": "d7de6027-43d3-48c7-8490-d0c290e1c761",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/single_epoch_baseline/group2_subject1.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group2_stats_files",
        #                     "file_name": "group2_subject2.STATISTICS.csv",
        #                     "file_id": "4e2f5699-552c-43f6-9aaf-bf52ab315015",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/single_epoch_baseline/group2_subject2.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_event_aligned_traces",
        #                     "file_name": "event_aligned_activity_DrugGroup.csv",
        #                     "file_id": "19bd0ddd-5d0d-41c9-a364-bb096cb2454f",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_DrugGroup.csv",
        #                     "file_type": "combined_peri_event_traces",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned population activity across epochs",
        #                             "help": "Comparison of event-aligned average population activity across the epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_DrugGroup.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned population activity",
        #                             "help": "Event-aligned average population activity line plot (epoch: Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_DrugGroup_Drug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap (epoch: Drug)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_heatmap_DrugGroup_Drug.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group1_event_aligned_statistics",
        #                     "file_name": "event_aligned_statistics_DrugGroup.csv",
        #                     "file_id": "6fe2765c-c0a9-4336-9217-74be259fc14c",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_statistics_DrugGroup.csv",
        #                     "file_type": "combined_peri_event_statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_by_modulation_DrugGroup_Drug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of up-modulated cells",
        #                             "help": "Comparison of event-aligned activity of up-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_up_modulated_DrugGroup.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of down-modulated cells",
        #                             "help": "Comparison of event-aligned activity of down-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_down_modulated_DrugGroup.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of non-modulated cells",
        #                             "help": "Comparison of event-aligned activity of non-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_non_modulated_DrugGroup.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in each sub-population (up-, down-, and non-modulated neurons) (epoch: Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/fraction_of_modulated_neurons_DrugGroup_Drug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Number of modulated cells per epoch",
        #                             "help": "Number of up-, down-, and non-modulated neurons per epoch.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/num_modulated_cells_per_epoch_DrugGroup.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Mean post-pre activity per epoch",
        #                             "help": "Comparison of mean post-pre activity across the epochs. The error bars represent the standard error of the mean.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/mean_post_minus_pre_activity_per_epoch_DrugGroup.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group2_event_aligned_traces",
        #                     "file_name": "event_aligned_activity_BaselineGroup.csv",
        #                     "file_id": "20adf726-4163-4abf-84b4-2065992df1cb",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_BaselineGroup.csv",
        #                     "file_type": "combined_peri_event_traces",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned population activity across epochs",
        #                             "help": "Comparison of event-aligned average population activity across the epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_BaselineGroup.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned population activity",
        #                             "help": "Event-aligned average population activity line plot (epoch: Baseline).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_BaselineGroup_Baseline.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap (epoch: Baseline)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_heatmap_BaselineGroup_Baseline.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group2_event_aligned_statistics",
        #                     "file_name": "event_aligned_statistics_BaselineGroup.csv",
        #                     "file_id": "d4fb1b42-16aa-443b-bc1d-944f6c3ac94e",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_statistics_BaselineGroup.csv",
        #                     "file_type": "combined_peri_event_statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Baseline).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_by_modulation_BaselineGroup_Baseline.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of up-modulated cells",
        #                             "help": "Comparison of event-aligned activity of up-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_up_modulated_BaselineGroup.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of down-modulated cells",
        #                             "help": "Comparison of event-aligned activity of down-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_down_modulated_BaselineGroup.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of non-modulated cells",
        #                             "help": "Comparison of event-aligned activity of non-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_non_modulated_BaselineGroup.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in each sub-population (up-, down-, and non-modulated neurons) (epoch: Baseline).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/fraction_of_modulated_neurons_BaselineGroup_Baseline.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Number of modulated cells per epoch",
        #                             "help": "Number of up-, down-, and non-modulated neurons per epoch.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/num_modulated_cells_per_epoch_BaselineGroup.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Mean post-pre activity per epoch",
        #                             "help": "Comparison of mean post-pre activity across the epochs. The error bars represent the standard error of the mean.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/mean_post_minus_pre_activity_per_epoch_BaselineGroup.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #             ],
        #         }
        #     ],
        # }
        # self.assertTrue(
        #     validate_output_manifest(
        #         expected_output_manifest=exp_manifest,
        #         tool_metadata=self.tool_metadata[self.tool_key],
        #         actual_manifest_data_filepath=self.output_manifest_file,
        #     )
        # )

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

        # validate output manifest
        # exp_manifest = {
        #     "schema_version": "2.0.0",
        #     "groups": [
        #         {
        #             "group_key": "combine_compare_peri_event_data_across_epochs_output",
        #             "group_type": "tool_output",
        #             "group_id": "99ee7721-c2cd-4888-a35d-c1e5bd8f511d",
        #             "series": [],
        #             "files": [
        #                 {
        #                     "file_key": "group1_traces_files",
        #                     "file_name": "group1_subject1.TRACES.csv",
        #                     "file_id": "83dd1b22-1fc6-4b55-b033-06882c24ddd3",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/single_epoch_baseline/group1_subject1.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_traces_files",
        #                     "file_name": "group1_subject2.TRACES.csv",
        #                     "file_id": "aa1bcf3e-6354-4cab-9167-167252fc0202",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/single_epoch_baseline/group1_subject2.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_traces_files",
        #                     "file_name": "group1_subject1.TRACES.csv",
        #                     "file_id": "b0f6b033-c85e-4a00-b82b-331b393c722d",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/single_epoch_drug/group1_subject1.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_traces_files",
        #                     "file_name": "group1_subject2.TRACES.csv",
        #                     "file_id": "e3fcd3e7-6626-4bc1-a8c1-c0967f9da19c",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/single_epoch_drug/group1_subject2.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_stats_files",
        #                     "file_name": "group1_subject1.STATISTICS.csv",
        #                     "file_id": "e42c1b53-24a6-43b0-8518-47f9c7d02abc",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/single_epoch_baseline/group1_subject1.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_stats_files",
        #                     "file_name": "group1_subject2.STATISTICS.csv",
        #                     "file_id": "4f2af689-0477-445c-b7e1-0d9e930aee20",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/single_epoch_baseline/group1_subject2.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_stats_files",
        #                     "file_name": "group1_subject1.STATISTICS.csv",
        #                     "file_id": "03ecd035-b408-41af-a6d9-d51fec1093d2",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/single_epoch_drug/group1_subject1.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_stats_files",
        #                     "file_name": "group1_subject2.STATISTICS.csv",
        #                     "file_id": "bb5b8a45-2481-4b4e-ad66-e1170ea690d9",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data_across_epochs/single_epoch_drug/group1_subject2.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_event_aligned_traces",
        #                     "file_name": "event_aligned_activity_group1.csv",
        #                     "file_id": "74b68b11-0d4e-462b-8df8-68e145fdc02e",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_group1.csv",
        #                     "file_type": "combined_peri_event_traces",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned population activity across epochs",
        #                             "help": "Comparison of event-aligned average population activity across the epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned population activity",
        #                             "help": "Event-aligned average population activity line plot (epoch: Baseline).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_group1_Baseline.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned population activity",
        #                             "help": "Event-aligned average population activity line plot (epoch: Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_population_activity_group1_Drug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap (epoch: Baseline)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_heatmap_group1_Baseline.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap (epoch: Drug)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_heatmap_group1_Drug.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group1_event_aligned_statistics",
        #                     "file_name": "event_aligned_statistics_group1.csv",
        #                     "file_id": "5460966e-4516-4ed2-a4d5-248b30738597",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_statistics_group1.csv",
        #                     "file_type": "combined_peri_event_statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Baseline).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_by_modulation_group1_Baseline.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_by_modulation_group1_Drug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of up-modulated cells",
        #                             "help": "Comparison of event-aligned activity of up-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_up_modulated_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of down-modulated cells",
        #                             "help": "Comparison of event-aligned activity of down-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_down_modulated_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned activity of non-modulated cells",
        #                             "help": "Comparison of event-aligned activity of non-modulated cells across epochs.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/event_aligned_activity_non_modulated_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in each sub-population (up-, down-, and non-modulated neurons) (epoch: Baseline).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/fraction_of_modulated_neurons_group1_Baseline.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in each sub-population (up-, down-, and non-modulated neurons) (epoch: Drug).",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/fraction_of_modulated_neurons_group1_Drug.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Number of modulated cells per epoch",
        #                             "help": "Number of up-, down-, and non-modulated neurons per epoch.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/num_modulated_cells_per_epoch_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Mean post-pre activity per epoch",
        #                             "help": "Comparison of mean post-pre activity across the epochs. The error bars represent the standard error of the mean.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/mean_post_minus_pre_activity_per_epoch_group1.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group1_epoch_comparison_data",
        #                     "file_name": "pairwise_epoch_comparisons_group1.csv",
        #                     "file_id": "c50ef0bc-2f5b-4db1-89c6-6f0188b19090",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/pairwise_epoch_comparisons_group1.csv",
        #                     "file_type": "peri_event_comparison_data",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "parent_ids": [
        #                         "74b68b11-0d4e-462b-8df8-68e145fdc02e",
        #                         "5460966e-4516-4ed2-a4d5-248b30738597",
        #                     ],
        #                     "preview": [
        #                         {
        #                             "name": "Post-pre distribution across the epochs",
        #                             "help": "Distribution of post-pre activity across epochs displayed using a box plot. Lines connect the same cells together.",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data_across_epochs/post_minus_pre_boxplot_group1.svg",
        #                             "file_format": "svg",
        #                         }
        #                     ],
        #                 },
        #             ],
        #         }
        #     ],
        # }
        # self.assertTrue(
        #     validate_output_manifest(
        #         expected_output_manifest=exp_manifest,
        #         tool_metadata=self.tool_metadata[self.tool_key],
        #         actual_manifest_data_filepath=self.output_manifest_file,
        #     )
        # )
