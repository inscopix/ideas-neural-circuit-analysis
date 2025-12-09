import os
import shutil
import unittest
import pandas as pd
from ideas.exceptions import IdeasError
from analysis.combine_compare_peri_event_data import (
    combine_compare_peri_event_data,
)


class TestCombineComparePeriEventData(unittest.TestCase):
    """Tests for the combine and compare peri-event analysis tool."""

    # tool info
    tool_key = "combine_compare_peri_event_data"

    # define directories
    temporary_dir = "/tmp"
    input_dir = "data/combine_compare_peri_event_data"
    output_dir = os.path.join(
        temporary_dir, "tmp_combine_compare_peri_event_data"
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

    # Valid Cases
    def test_combine_single_group(self):
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
        combine_compare_peri_event_data(
            group1_traces_files=group1_traces_files,
            group1_stats_files=group1_stats_files,
            group1_name=group_name,
            group2_traces_files=[],
            group2_stats_files=[],
            group2_name="",
            comparison_type=None,
            data_pairing=None,
            significance_threshold=None,
            average_method="neurons",
            tolerance=0.01,
            output_dir=self.output_dir,
        )

        # validate existence of expected output files
        actual_files = os.listdir(self.output_dir)
        expected_files = [
            # "output_manifest.json",
            "output_metadata.json",
            f"event_aligned_activity_{group_name}.csv",
            f"event_aligned_activity_by_modulation_{group_name}.preview.svg",
            f"fraction_of_modulated_neurons_{group_name}.preview.svg",
            f"event_aligned_statistics_{group_name}.csv",
            f"event_aligned_population_activity_{group_name}.preview.svg",
            f"event_aligned_single_cell_activity_heatmap_{group_name}.preview.svg",
        ]
        self.assertEqual(sorted(actual_files), sorted(expected_files))

        # validate event-aligned traces
        df_traces = pd.read_csv(
            os.path.join(
                self.output_dir, f"event_aligned_activity_{group_name}.csv"
            )
        )
        self.assertEqual(df_traces.shape, (119, 249))

        # validate population mean and sem (the expected values were computed manually)
        self.assertAlmostEqual(
            -0.00139902927655282, df_traces["population_mean"][0]
        )
        self.assertAlmostEqual(
            0.00661793085084028, df_traces["population_sem"][0]
        )

        # validate sub-population mean and sem
        self.assertAlmostEqual(
            -0.049593321516071064, df_traces["up_modulated_mean"][0]
        )
        self.assertAlmostEqual(
            0.02248465911945461, df_traces["up_modulated_sem"][0]
        )
        self.assertAlmostEqual(
            0.0284092191640714, df_traces["down_modulated_mean"][0]
        )
        self.assertAlmostEqual(
            0.0325825083730468, df_traces["down_modulated_sem"][0]
        )
        self.assertAlmostEqual(
            0.000372933333441427, df_traces["non_modulated_mean"][0]
        )
        self.assertAlmostEqual(
            0.006994805756296599, df_traces["non_modulated_sem"][0]
        )

        # validate event-aligned statistics
        df_stats = pd.read_csv(
            os.path.join(
                self.output_dir, f"event_aligned_statistics_{group_name}.csv"
            )
        )
        self.assertEqual(df_stats.shape, (240, 8))

        expected_cols = [
            "name",
            "true_mean_post-pre",
            "shuffled_mean_post-pre",
            "shuffled_std_post-pre",
            "z-score",
            "p-value",
            "modulation",
            "file",
        ]
        actual_cols = df_stats.columns.tolist()
        self.assertEqual(actual_cols, expected_cols)

        # ensure the number of modulated cells match the input
        # since we used the same p-value to reclassify them (p=0.05)
        self.assertEqual(
            7,
            len(
                df_stats[
                    (df_stats.modulation == 1.0)
                    & (df_stats.file == "group1_subject1.STATISTICS.csv")
                ]
            ),
        )
        self.assertEqual(
            6,
            len(
                df_stats[
                    (df_stats.modulation == -1.0)
                    & (df_stats.file == "group1_subject1.STATISTICS.csv")
                ]
            ),
        )
        self.assertEqual(
            107,
            len(
                df_stats[
                    (df_stats.modulation == 0.0)
                    & (df_stats.file == "group1_subject1.STATISTICS.csv")
                ]
            ),
        )

        self.assertEqual(
            6,
            len(
                df_stats[
                    (df_stats.modulation == 1.0)
                    & (df_stats.file == "group1_subject2.STATISTICS.csv")
                ]
            ),
        )
        self.assertEqual(
            2,
            len(
                df_stats[
                    (df_stats.modulation == -1.0)
                    & (df_stats.file == "group1_subject2.STATISTICS.csv")
                ]
            ),
        )
        self.assertEqual(
            112,
            len(
                df_stats[
                    (df_stats.modulation == 0.0)
                    & (df_stats.file == "group1_subject2.STATISTICS.csv")
                ]
            ),
        )

        # validate output manifest
        # expected_output_manifest = {
        #     "schema_version": "2.0.0",
        #     "groups": [
        #         {
        #             "group_key": "combine_compare_peri_event_data_output",
        #             "group_type": "tool_output",
        #             "group_id": "ba34b173-353d-45b4-9dfa-8e4ddab00090",
        #             "series": [],
        #             "files": [
        #                 {
        #                     "file_key": "group1_traces_files",
        #                     "file_name": "group1_subject1.TRACES.csv",
        #                     "file_id": "28b5fa10-f3c3-43cd-87b2-da6b4f3fbbfa",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare"
        #                     "_peri_event_data/group1/group1_subject1"
        #                     ".TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_traces_files",
        #                     "file_name": "group1_subject2.TRACES.csv",
        #                     "file_id": "4252298c-eb35-45bf-91d9-58fba8841501",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare"
        #                     "_peri_event_data/group1/group1_subject2"
        #                     ".TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_stats_files",
        #                     "file_name": "group1_subject1.STATISTICS.csv",
        #                     "file_id": "240a79cf-7872-4fd7-a687-7710667adb9a",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare"
        #                     "_peri_event_data/group1/group1_subject1"
        #                     ".STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_stats_files",
        #                     "file_name": "group1_subject2.STATISTICS.csv",
        #                     "file_id": "578d0ee6-525d-4a87-b9f3-96a044938977",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_"
        #                     "compare_peri_event_data/group1/group1"
        #                     "_subject2.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_event_aligned_activity_"
        #                     "traces_event_aligned_activity_traces"
        #                     "_csv_file",
        #                     "file_name": "event_aligned_activity_group1.csv",
        #                     "file_id": "4150ca1f-90e0-469f-ba71-d4b509d40560",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                     "event_aligned_activity_group1.csv",
        #                     "file_type": "combined_peri_event_traces",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned population activity figure",
        #                             "help": "Event-aligned average population "
        #                             "activity line plot",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "event_aligned_population_activity_group1"
        #                             ".preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "event_aligned_single_cell_activity_heatmap"
        #                             "_group1.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group1_event_aligned_statistics_event_"
        #                     "aligned_activity_statistics_csv_file",
        #                     "file_name": "event_aligned_statistics_group1.csv",
        #                     "file_id": "76a0a431-3d02-45ca-a21a-64244e9ed5c9",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                     "event_aligned_statistics_group1.csv",
        #                     "file_type": "combined_peri_event_statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity "
        #                             "line plot (up-, down-, and non-modulated neurons)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "event_aligned_activity_by_modulation_group1"
        #                             ".preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in"
        #                             " each sub-population (up-, down-, and non-modulated "
        #                             "neurons)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "fraction_of_modulated_neurons_group1"
        #                             ".preview.svg",
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
        #         expected_output_manifest=expected_output_manifest,
        #         tool_metadata=self.tool_metadata[self.tool_key],
        #         actual_manifest_data_filepath=self.output_manifest_file,
        #     )
        # )

    def test_combine_single_group_recordings_average_method(self):
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
        combine_compare_peri_event_data(
            group1_traces_files=group1_traces_files,
            group1_stats_files=group1_stats_files,
            group1_name=group_name,
            group2_traces_files=[],
            group2_stats_files=[],
            group2_name="",
            comparison_type="two_tailed",
            data_pairing="unpaired",
            significance_threshold=None,
            average_method="recordings",
            tolerance=0.01,
            output_dir=self.output_dir,
        )

        # validate existence of expected output files
        actual_files = os.listdir(self.output_dir)
        expected_files = [
            # "output_manifest.json",
            "output_metadata.json",
            f"event_aligned_activity_{group_name}.csv",
            f"event_aligned_activity_by_modulation_{group_name}.preview.svg",
            f"fraction_of_modulated_neurons_{group_name}.preview.svg",
            f"event_aligned_statistics_{group_name}.csv",
            f"event_aligned_population_activity_{group_name}.preview.svg",
            f"event_aligned_single_cell_activity_heatmap_{group_name}.preview.svg",
        ]
        self.assertEqual(sorted(actual_files), sorted(expected_files))

        # validate event-aligned traces
        df_traces = pd.read_csv(
            os.path.join(
                self.output_dir, f"event_aligned_activity_{group_name}.csv"
            )
        )
        self.assertEqual(df_traces.shape, (119, 249))

        # validate population mean and sem (the expected values were computed manually)
        self.assertAlmostEqual(
            -0.00139902927655283, df_traces["population_mean"][0]
        )
        self.assertAlmostEqual(
            0.00033011431246619, df_traces["population_sem"][0]
        )

        # validate sub-population mean and sem
        self.assertAlmostEqual(
            -0.051186460829229084, df_traces["up_modulated_mean"][0]
        )
        self.assertAlmostEqual(
            0.02071081107105348, df_traces["up_modulated_sem"][0]
        )
        self.assertAlmostEqual(
            0.0414322152247661, df_traces["down_modulated_mean"][0]
        )
        self.assertAlmostEqual(
            0.0260459921213894, df_traces["down_modulated_sem"][0]
        )
        self.assertAlmostEqual(
            0.000363094912905372, df_traces["non_modulated_mean"][0]
        )
        self.assertAlmostEqual(
            0.000430922819479477, df_traces["non_modulated_sem"][0]
        )

        # validate event-aligned statistics
        df_stats = pd.read_csv(
            os.path.join(
                self.output_dir, f"event_aligned_statistics_{group_name}.csv"
            )
        )
        self.assertEqual(df_stats.shape, (240, 8))

        expected_cols = [
            "name",
            "true_mean_post-pre",
            "shuffled_mean_post-pre",
            "shuffled_std_post-pre",
            "z-score",
            "p-value",
            "modulation",
            "file",
        ]
        actual_cols = df_stats.columns.tolist()
        self.assertEqual(actual_cols, expected_cols)

        # ensure the number of modulated cells match the input
        # since we used the same p-value to reclassify them (p=0.05)
        self.assertEqual(
            7,
            len(
                df_stats[
                    (df_stats.modulation == 1.0)
                    & (df_stats.file == "group1_subject1.STATISTICS.csv")
                ]
            ),
        )
        self.assertEqual(
            6,
            len(
                df_stats[
                    (df_stats.modulation == -1.0)
                    & (df_stats.file == "group1_subject1.STATISTICS.csv")
                ]
            ),
        )
        self.assertEqual(
            107,
            len(
                df_stats[
                    (df_stats.modulation == 0.0)
                    & (df_stats.file == "group1_subject1.STATISTICS.csv")
                ]
            ),
        )

        self.assertEqual(
            6,
            len(
                df_stats[
                    (df_stats.modulation == 1.0)
                    & (df_stats.file == "group1_subject2.STATISTICS.csv")
                ]
            ),
        )
        self.assertEqual(
            2,
            len(
                df_stats[
                    (df_stats.modulation == -1.0)
                    & (df_stats.file == "group1_subject2.STATISTICS.csv")
                ]
            ),
        )
        self.assertEqual(
            112,
            len(
                df_stats[
                    (df_stats.modulation == 0.0)
                    & (df_stats.file == "group1_subject2.STATISTICS.csv")
                ]
            ),
        )

        # validate output manifest
        # expected_output_manifest = {
        #     "schema_version": "2.0.0",
        #     "groups": [
        #         {
        #             "group_key": "combine_compare_peri_event_data_output",
        #             "group_type": "tool_output",
        #             "group_id": "ba34b173-353d-45b4-9dfa-8e4ddab00090",
        #             "series": [],
        #             "files": [
        #                 {
        #                     "file_key": "group1_traces_files",
        #                     "file_name": "group1_subject1.TRACES.csv",
        #                     "file_id": "28b5fa10-f3c3-43cd-87b2-da6b4f3fbbfa",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare"
        #                     "_peri_event_data/group1/group1_subject1"
        #                     ".TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_traces_files",
        #                     "file_name": "group1_subject2.TRACES.csv",
        #                     "file_id": "4252298c-eb35-45bf-91d9-58fba8841501",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_"
        #                     "compare_peri_event_data/group1/group1"
        #                     "_subject2.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_stats_files",
        #                     "file_name": "group1_subject1.STATISTICS.csv",
        #                     "file_id": "240a79cf-7872-4fd7-a687-7710667adb9a",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare"
        #                     "_peri_event_data/group1/group1_subject1"
        #                     ".STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_stats_files",
        #                     "file_name": "group1_subject2.STATISTICS.csv",
        #                     "file_id": "578d0ee6-525d-4a87-b9f3-96a044938977",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare"
        #                     "_peri_event_data/group1/group1_subject2."
        #                     "STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_event_aligned_activity_traces_"
        #                     "event_aligned_activity_traces_csv_file",
        #                     "file_name": "event_aligned_activity_group1.csv",
        #                     "file_id": "4150ca1f-90e0-469f-ba71-d4b509d40560",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                     "event_aligned_activity_group1.csv",
        #                     "file_type": "combined_peri_event_traces",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned population activity figure",
        #                             "help": "Event-aligned average population"
        #                             " activity line plot",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_"
        #                             "event_data/event_aligned_population"
        #                             "_activity_group1.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data"
        #                             "/event_aligned_single_cell_activity_heatmap"
        #                             "_group1.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group1_event_aligned_statistics_event_"
        #                     "aligned_activity_statistics_csv_file",
        #                     "file_name": "event_aligned_statistics_group1.csv",
        #                     "file_id": "76a0a431-3d02-45ca-a21a-64244e9ed5c9",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                     "event_aligned_statistics_group1.csv",
        #                     "file_type": "combined_peri_event_statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity "
        #                             "line plot (up-, down-, and non-modulated neurons)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "event_aligned_activity_by_modulation_group1"
        #                             ".preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in "
        #                             "each sub-population (up-, down-, and non-modulated "
        #                             "neurons)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "fraction_of_modulated_neurons_group"
        #                             "1.preview.svg",
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
        #         expected_output_manifest=expected_output_manifest,
        #         tool_metadata=self.tool_metadata[self.tool_key],
        #         actual_manifest_data_filepath=self.output_manifest_file,
        #     )
        # )

    def test_combine_and_compare_two_multi_file_groups(self):
        # define input files
        group1_name = "group1"
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

        group2_name = "group2"
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
        combine_compare_peri_event_data(
            group1_traces_files=group1_traces_files,
            group1_stats_files=group1_stats_files,
            group1_name=group1_name,
            group2_traces_files=group2_traces_files,
            group2_stats_files=group2_stats_files,
            group2_name=group2_name,
            comparison_type="two_tailed",
            data_pairing="paired",
            significance_threshold=None,
            average_method="neurons",
            tolerance=0.01,
            output_dir=self.output_dir,
        )

        # validate existence of expected output files
        actual_files = os.listdir(self.output_dir)
        expected_files = [
            # output files expected by the system
            # "output_manifest.json",
            "output_metadata.json",
            # output files from combining data in group 1
            f"event_aligned_activity_{group1_name}.csv",
            f"event_aligned_activity_by_modulation_{group1_name}.preview.svg",
            f"fraction_of_modulated_neurons_{group1_name}.preview.svg",
            f"event_aligned_statistics_{group1_name}.csv",
            f"event_aligned_population_activity_{group1_name}.preview.svg",
            f"event_aligned_single_cell_activity_heatmap_{group1_name}.preview.svg",
            # output files from combining data in group 2
            f"event_aligned_activity_{group2_name}.csv",
            f"event_aligned_activity_by_modulation_{group2_name}.preview.svg",
            f"fraction_of_modulated_neurons_{group2_name}.preview.svg",
            f"event_aligned_statistics_{group2_name}.csv",
            f"event_aligned_population_activity_{group2_name}.preview.svg",
            f"event_aligned_single_cell_activity_heatmap_{group2_name}.preview.svg",
            # comparison output files
            "comparison_data.csv",
            "event_aligned_population_activity_comparison.preview.svg",
            "event_aligned_subpopulation_activity_comparison.preview.svg",
            "modulated_fractions_comparison.preview.svg",
        ]
        self.assertEqual(sorted(actual_files), sorted(expected_files))

        # validate event-aligned traces for GROUP 1
        df_traces_group1 = pd.read_csv(
            os.path.join(
                self.output_dir, f"event_aligned_activity_{group1_name}.csv"
            )
        )
        self.assertEqual(df_traces_group1.shape, (119, 249))

        # validate population mean and sem (the expected values were computed manually)
        self.assertAlmostEqual(
            -0.00139902927655282, df_traces_group1["population_mean"][0]
        )
        self.assertAlmostEqual(
            0.00661793085084028, df_traces_group1["population_sem"][0]
        )

        # validate sub-population mean and sem
        self.assertAlmostEqual(
            -0.049593321516071064, df_traces_group1["up_modulated_mean"][0]
        )
        self.assertAlmostEqual(
            0.0284092191640714, df_traces_group1["down_modulated_mean"][0]
        )
        self.assertAlmostEqual(
            0.000372933333441427, df_traces_group1["non_modulated_mean"][0]
        )

        # validate event-aligned statistics for GROUP 1
        df_stats_group1 = pd.read_csv(
            os.path.join(
                self.output_dir, f"event_aligned_statistics_{group1_name}.csv"
            )
        )
        self.assertEqual(df_stats_group1.shape, (240, 8))

        expected_cols = [
            "name",
            "true_mean_post-pre",
            "shuffled_mean_post-pre",
            "shuffled_std_post-pre",
            "z-score",
            "p-value",
            "modulation",
            "file",
        ]
        actual_cols = df_stats_group1.columns.tolist()
        self.assertEqual(actual_cols, expected_cols)

        # ensure the number of modulated cells match the input
        # since we used the same p-value to reclassify them (p=0.05)
        self.assertEqual(
            7,
            len(
                df_stats_group1[
                    (df_stats_group1.modulation == 1.0)
                    & (
                        df_stats_group1.file
                        == "group1_subject1.STATISTICS.csv"
                    )
                ]
            ),
        )
        self.assertEqual(
            6,
            len(
                df_stats_group1[
                    (df_stats_group1.modulation == -1.0)
                    & (
                        df_stats_group1.file
                        == "group1_subject1.STATISTICS.csv"
                    )
                ]
            ),
        )
        self.assertEqual(
            107,
            len(
                df_stats_group1[
                    (df_stats_group1.modulation == 0.0)
                    & (
                        df_stats_group1.file
                        == "group1_subject1.STATISTICS.csv"
                    )
                ]
            ),
        )

        self.assertEqual(
            6,
            len(
                df_stats_group1[
                    (df_stats_group1.modulation == 1.0)
                    & (
                        df_stats_group1.file
                        == "group1_subject2.STATISTICS.csv"
                    )
                ]
            ),
        )
        self.assertEqual(
            2,
            len(
                df_stats_group1[
                    (df_stats_group1.modulation == -1.0)
                    & (
                        df_stats_group1.file
                        == "group1_subject2.STATISTICS.csv"
                    )
                ]
            ),
        )
        self.assertEqual(
            112,
            len(
                df_stats_group1[
                    (df_stats_group1.modulation == 0.0)
                    & (
                        df_stats_group1.file
                        == "group1_subject2.STATISTICS.csv"
                    )
                ]
            ),
        )

        # validate event-aligned traces for GROUP 2
        df_traces_group2 = pd.read_csv(
            os.path.join(
                self.output_dir, f"event_aligned_activity_{group2_name}.csv"
            )
        )
        self.assertEqual(df_traces_group2.shape, (119, 249))

        # validate population mean and sem (the expected values were computed manually)
        self.assertAlmostEqual(
            -0.0278609511299807, df_traces_group2["population_mean"][0]
        )
        self.assertAlmostEqual(
            0.00694676565981122, df_traces_group2["population_sem"][0]
        )

        # validate sub-population mean and sem
        self.assertAlmostEqual(
            -0.10627978328120656, df_traces_group2["up_modulated_mean"][0]
        )
        self.assertAlmostEqual(
            0.09017987368724932, df_traces_group2["down_modulated_mean"][0]
        )
        self.assertAlmostEqual(
            -0.02652140411247937, df_traces_group2["non_modulated_mean"][0]
        )

        # validate event-aligned statistics for GROUP 2
        df_stats_group2 = pd.read_csv(
            os.path.join(
                self.output_dir, f"event_aligned_statistics_{group2_name}.csv"
            )
        )
        self.assertEqual(df_stats_group2.shape, (240, 8))

        expected_cols = [
            "name",
            "true_mean_post-pre",
            "shuffled_mean_post-pre",
            "shuffled_std_post-pre",
            "z-score",
            "p-value",
            "modulation",
            "file",
        ]
        actual_cols = df_stats_group2.columns.tolist()
        self.assertEqual(actual_cols, expected_cols)

        # ensure the number of modulated cells match the input
        # since we used the same p-value to reclassify them (p=0.05)
        self.assertEqual(
            23,
            len(
                df_stats_group2[
                    (df_stats_group2.modulation == 1.0)
                    & (
                        df_stats_group2.file
                        == "group2_subject1.STATISTICS.csv"
                    )
                ]
            ),
        )
        self.assertEqual(
            18,
            len(
                df_stats_group2[
                    (df_stats_group2.modulation == -1.0)
                    & (
                        df_stats_group2.file
                        == "group2_subject1.STATISTICS.csv"
                    )
                ]
            ),
        )
        self.assertEqual(
            79,
            len(
                df_stats_group2[
                    (df_stats_group2.modulation == 0.0)
                    & (
                        df_stats_group2.file
                        == "group2_subject1.STATISTICS.csv"
                    )
                ]
            ),
        )

        self.assertEqual(
            22,
            len(
                df_stats_group2[
                    (df_stats_group2.modulation == 1.0)
                    & (
                        df_stats_group2.file
                        == "group2_subject2.STATISTICS.csv"
                    )
                ]
            ),
        )
        self.assertEqual(
            10,
            len(
                df_stats_group2[
                    (df_stats_group2.modulation == -1.0)
                    & (
                        df_stats_group2.file
                        == "group2_subject2.STATISTICS.csv"
                    )
                ]
            ),
        )
        self.assertEqual(
            88,
            len(
                df_stats_group2[
                    (df_stats_group2.modulation == 0.0)
                    & (
                        df_stats_group2.file
                        == "group2_subject2.STATISTICS.csv"
                    )
                ]
            ),
        )

        # validate output manifest
        # expected_output_manifest = {
        #     "schema_version": "2.0.0",
        #     "groups": [
        #         {
        #             "group_key": "combine_compare_peri_event_data_output",
        #             "group_type": "tool_output",
        #             "group_id": "7765d844-71e1-4203-af4c-3a4134dab37f",
        #             "series": [],
        #             "files": [
        #                 {
        #                     "file_key": "group1_traces_files",
        #                     "file_name": "group1_subject1.TRACES.csv",
        #                     "file_id": "845e3522-91fe-459f-b05a-fafb15a93aa4",
        #                     "file_path": "/ideas/toolbox/tests/data/"
        #                     "combine_compare_peri_event_data/"
        #                     "group1/group1_subject1.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_traces_files",
        #                     "file_name": "group1_subject2.TRACES.csv",
        #                     "file_id": "9ef2da72-5622-4c07-90dd-5f59797ffaf5",
        #                     "file_path": "/ideas/toolbox/tests/data/"
        #                     "combine_compare_peri_event_data/"
        #                     "group1/group1_subject2.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_stats_files",
        #                     "file_name": "group1_subject1.STATISTICS.csv",
        #                     "file_id": "504b21d5-5475-4447-ad13-5499da4834e3",
        #                     "file_path": "/ideas/toolbox/tests/data/"
        #                     "combine_compare_peri_event_data/"
        #                     "group1/group1_subject1.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_stats_files",
        #                     "file_name": "group1_subject2.STATISTICS.csv",
        #                     "file_id": "5a3749b8-8af2-454b-9172-2eccb122ec11",
        #                     "file_path": "/ideas/toolbox/tests/data/"
        #                     "combine_compare_peri_event_data/"
        #                     "group1/group1_subject2.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group2_traces_files",
        #                     "file_name": "group2_subject1.TRACES.csv",
        #                     "file_id": "bc9d47f8-d8e9-452d-8de1-480eca7169aa",
        #                     "file_path": "/ideas/toolbox/tests/data/"
        #                     "combine_compare_peri_event_data/"
        #                     "group2/group2_subject1.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group2_traces_files",
        #                     "file_name": "group2_subject2.TRACES.csv",
        #                     "file_id": "8708f587-9c88-414a-982c-07774bfed02f",
        #                     "file_path": "/ideas/toolbox/tests/data/"
        #                     "combine_compare_peri_event_data/"
        #                     "group2/group2_subject2.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group2_stats_files",
        #                     "file_name": "group2_subject1.STATISTICS.csv",
        #                     "file_id": "33ee110c-307f-44ad-8173-ad0f43cfac3c",
        #                     "file_path": "/ideas/toolbox/tests/data/"
        #                     "combine_compare_peri_event_data/"
        #                     "group2/group2_subject1.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group2_stats_files",
        #                     "file_name": "group2_subject2.STATISTICS.csv",
        #                     "file_id": "0dbcff3e-b1a4-483a-96b8-509e603c48c5",
        #                     "file_path": "/ideas/toolbox/tests/data/"
        #                     "combine_compare_peri_event_data/"
        #                     "group2/group2_subject2.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_event_aligned_activity_traces_"
        #                     "event_aligned_activity_traces_csv_file",
        #                     "file_name": "event_aligned_activity_group1.csv",
        #                     "file_id": "05f93015-d371-47ae-b838-dc29ec44cf7e",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                     "event_aligned_activity_group1.csv",
        #                     "file_type": "combined_peri_event_traces",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned population activity figure",
        #                             "help": "Event-aligned average population activity line plot",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "event_aligned_population_activity"
        #                             "_group1.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "event_aligned_single_cell_activity_heatmap"
        #                             "_group1.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group1_event_aligned_statistics_event_"
        #                     "aligned_activity_statistics_csv_file",
        #                     "file_name": "event_aligned_statistics_group1.csv",
        #                     "file_id": "ed5ad3d9-fe28-492e-8c7b-3ae432fd34a7",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                     "event_aligned_statistics_group1.csv",
        #                     "file_type": "combined_peri_event_statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity "
        #                             "line plot (up-, down-, and non-modulated neurons)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "event_aligned_activity_by_modulation_group1"
        #                             ".preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in "
        #                             "each sub-population (up-, down-, and non-modulated"
        #                             " neurons)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "fraction_of_modulated_neurons_group1.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group2_event_aligned_activity_traces_event"
        #                     "_aligned_activity_traces_csv_file",
        #                     "file_name": "event_aligned_activity_group2.csv",
        #                     "file_id": "23835d2d-56c1-4a54-b5c9-0ff9df0f199b",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                     "event_aligned_activity_group2.csv",
        #                     "file_type": "combined_peri_event_traces",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned population activity figure",
        #                             "help": "Event-aligned average population activity line plot",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "event_aligned_population_activity_group2"
        #                             ".preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "event_aligned_single_cell_activity_heatmap"
        #                             "_group2.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group2_event_aligned_statistics_event"
        #                     "_aligned_activity_statistics_csv_file",
        #                     "file_name": "event_aligned_statistics_group2.csv",
        #                     "file_id": "0dba3dc1-0364-4851-b2d8-7fe0a9577a7e",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                     "event_aligned_statistics_group2.csv",
        #                     "file_type": "combined_peri_event_statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity "
        #                             "line plot (up-, down-, and non-modulated neurons)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "event_aligned_activity_by_modulation_group2"
        #                             ".preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in "
        #                             "each sub-population (up-, down-, and non-modulated"
        #                             " neurons)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "fraction_of_modulated_neurons_group2.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "comparison_data_comparison_data_csv_file",
        #                     "file_name": "comparison_data.csv",
        #                     "file_id": "a6b3f95b-aca6-42db-9f1f-81a6f9adfbd1",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                     "comparison_data.csv",
        #                     "file_type": "peri_event_comparison_data",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned population activity figure",
        #                             "help": "Event-aligned average population activity line plot",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "event_aligned_population_activity_"
        #                             "comparison.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity "
        #                             "line plot",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "event_aligned_subpopulation_activity_comparison"
        #                             ".preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each modulation group",
        #                             "help": "Fraction of neurons in each modulation group",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "modulated_fractions_comparison.preview.svg",
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
        #         expected_output_manifest=expected_output_manifest,
        #         tool_metadata=self.tool_metadata[self.tool_key],
        #         actual_manifest_data_filepath=self.output_manifest_file,
        #     )
        # )

    def test_combine_and_compare_two_multi_file_groups_unequal_num_cells(self):
        # define input files
        group1_name = "Vehicle"
        group1_traces_files = [
            os.path.join(self.input_dir, f)
            for f in [
                "unequal_number_of_cells/group1_subject1.TRACES.csv",
                "unequal_number_of_cells/group1_subject2.TRACES.csv",
            ]
        ]
        group1_stats_files = [
            os.path.join(self.input_dir, f)
            for f in [
                "unequal_number_of_cells/group1_subject1.STATISTICS.csv",
                "unequal_number_of_cells/group1_subject2.STATISTICS.csv",
            ]
        ]

        group2_name = "Drug"
        group2_traces_files = [
            os.path.join(self.input_dir, f)
            for f in [
                "unequal_number_of_cells/group2_subject1.TRACES.csv",
                "unequal_number_of_cells/group2_subject2.TRACES.csv",
            ]
        ]
        group2_stats_files = [
            os.path.join(self.input_dir, f)
            for f in [
                "unequal_number_of_cells/group2_subject1.STATISTICS.csv",
                "unequal_number_of_cells/group2_subject2.STATISTICS.csv",
            ]
        ]

        # combine data
        combine_compare_peri_event_data(
            group1_traces_files=group1_traces_files,
            group1_stats_files=group1_stats_files,
            group1_name=group1_name,
            group2_traces_files=group2_traces_files,
            group2_stats_files=group2_stats_files,
            group2_name=group2_name,
            comparison_type="two_tailed",
            data_pairing="paired",
            significance_threshold=0.05,
            average_method="neurons",
            tolerance=0.01,
            output_dir=self.output_dir,
        )

        # validate existence of expected output files
        actual_files = os.listdir(self.output_dir)
        expected_files = [
            # output files expected by the system
            # "output_manifest.json",
            "output_metadata.json",
            # output files from combining data in group 1
            f"event_aligned_activity_{group1_name}.csv",
            f"event_aligned_activity_by_modulation_{group1_name}.preview.svg",
            f"fraction_of_modulated_neurons_{group1_name}.preview.svg",
            f"event_aligned_statistics_{group1_name}.csv",
            f"event_aligned_population_activity_{group1_name}.preview.svg",
            f"event_aligned_single_cell_activity_heatmap_{group1_name}.preview.svg",
            # output files from combining data in group 2
            f"event_aligned_activity_{group2_name}.csv",
            f"event_aligned_activity_by_modulation_{group2_name}.preview.svg",
            f"fraction_of_modulated_neurons_{group2_name}.preview.svg",
            f"event_aligned_statistics_{group2_name}.csv",
            f"event_aligned_population_activity_{group2_name}.preview.svg",
            f"event_aligned_single_cell_activity_heatmap_{group2_name}.preview.svg",
            # comparison output files
            "comparison_data.csv",
            "event_aligned_population_activity_comparison.preview.svg",
            "event_aligned_subpopulation_activity_comparison.preview.svg",
            "modulated_fractions_comparison.preview.svg",
        ]
        self.assertEqual(sorted(actual_files), sorted(expected_files))

    # Runtime Error Cases
    # Error Cases due to Invalid Inputs
    def test_combine_and_compare_invalid_data_type_for_group1(self):
        for group1_traces_files, group1_stats_files in [
            ("group1_traces.csv", ["group1_stats.csv"]),
            (["group1_traces.csv"], "group1_stats.csv"),
            ("group1_traces.csv", "group1_stats.csv"),
        ]:
            self.assertRaisesRegex(
                IdeasError,
                "The first input group must consist of a list of files.",
                combine_compare_peri_event_data,
                group1_traces_files=group1_traces_files,
                group1_stats_files=group1_stats_files,
                group1_name="Group 1",
                group2_traces_files=["group2_traces.csv"],
                group2_stats_files=["group2_stats.csv"],
                group2_name="Group 2",
                comparison_type="two_tailed",
                data_pairing="unpaired",
                significance_threshold=0.05,
                average_method="neurons",
                tolerance=0.01,
            )

    def test_combine_and_compare_insufficient_data_for_group1(self):
        for group1_traces_files, group1_stats_files in [
            (["group1_traces.csv"], []),
            ([], ["group1_stats.csv"]),
            ([], []),
        ]:
            self.assertRaisesRegex(
                IdeasError,
                "The first input group must contain at least two peri-event analyses.",
                combine_compare_peri_event_data,
                group1_traces_files=group1_traces_files,
                group1_stats_files=group1_stats_files,
                group1_name="Group 1",
                group2_traces_files=["group2_traces.csv"],
                group2_stats_files=["group2_stats.csv"],
                group2_name="Group 2",
                comparison_type="two_tailed",
                data_pairing="unpaired",
                significance_threshold=0.05,
                average_method="neurons",
                tolerance=0.01,
            )

    def test_combine_and_compare_unmatched_inputs_for_group1(self):
        for group1_traces_files, group1_stats_files in [
            (
                [
                    "group1_tracesA.csv",
                    "group1_tracesB.csv",
                    "group1_tracesC.csv",
                ],
                ["group1_statsA.csv", "group1_statsB.csv"],
            ),
            (
                ["group1_tracesA.csv", "group1_tracesB.csv"],
                [
                    "group1_statsA.csv",
                    "group1_statsB.csv",
                    "group1_statsC.csv",
                ],
            ),
        ]:
            self.assertRaisesRegex(
                IdeasError,
                "The number of traces and statistics files must match for the first input group.",
                combine_compare_peri_event_data,
                group1_traces_files=group1_traces_files,
                group1_stats_files=group1_stats_files,
                group1_name="Group 1",
                group2_traces_files=[
                    "group2_traces.csv",
                    "group2B_traces.csv",
                ],
                group2_stats_files=["group2_stats.csv", "group2B_stats.csv"],
                group2_name="Group 2",
                comparison_type="two_tailed",
                data_pairing="unpaired",
                significance_threshold=0.05,
                average_method="neurons",
                tolerance=0.01,
            )

    def test_combine_and_compare_unmatched_inputs_for_group2(self):
        for group2_traces_files, group2_stats_files in [
            (
                [
                    "group2_tracesA.csv",
                    "group2_tracesB.csv",
                    "group2_tracesC.csv",
                ],
                ["group2_statsA.csv", "group2_statsB.csv"],
            ),
            (
                ["group2_tracesA.csv", "group2_tracesB.csv"],
                [
                    "group2_statsA.csv",
                    "group2_statsB.csv",
                    "group2_statsC.csv",
                ],
            ),
        ]:
            self.assertRaisesRegex(
                IdeasError,
                "The number of traces and statistics files must match for the second input group.",
                combine_compare_peri_event_data,
                group1_traces_files=[
                    "group1_traces.csv",
                    "group1B_traces.csv",
                ],
                group1_stats_files=["group1_stats.csv", "group1B_stats.csv"],
                group1_name="Group 1",
                group2_traces_files=group2_traces_files,
                group2_stats_files=group2_stats_files,
                group2_name="Group 2",
                comparison_type="two_tailed",
                data_pairing="unpaired",
                significance_threshold=0.05,
                average_method="neurons",
                tolerance=0.01,
            )

    def test_combine_and_compare_missing_second_group_name(self):
        group1_name, group2_name = "GroupA", None
        self.assertRaisesRegex(
            IdeasError,
            "The second input group must have a group name.",
            combine_compare_peri_event_data,
            group1_traces_files=[
                "group1_traces.csv",
                "group1B_traces.csv",
            ],
            group1_stats_files=["group1_stats.csv", "group1B_stats.csv"],
            group1_name=group1_name,
            group2_traces_files=[
                "group2_traces.csv",
                "group2B_traces.csv",
            ],
            group2_stats_files=["group2_stats.csv", "group2B_stats.csv"],
            group2_name=group2_name,
            comparison_type="two_tailed",
            data_pairing="unpaired",
            significance_threshold=0.05,
            average_method="neurons",
            tolerance=0.01,
        )

    def test_combine_and_compare_identical_group_names(self):
        for group1_name, group2_name in [
            ("GroupA", "GroupA"),
            ("", ""),
        ]:
            self.assertRaisesRegex(
                IdeasError,
                "Group names cannot be identical.",
                combine_compare_peri_event_data,
                group1_traces_files=[
                    "group1_traces.csv",
                    "group1B_traces.csv",
                ],
                group1_stats_files=["group1_stats.csv", "group1B_stats.csv"],
                group1_name=group1_name,
                group2_traces_files=[
                    "group2_traces.csv",
                    "group2B_traces.csv",
                ],
                group2_stats_files=["group2_stats.csv", "group2B_stats.csv"],
                group2_name=group2_name,
                comparison_type="two_tailed",
                data_pairing="unpaired",
                significance_threshold=0.05,
                average_method="neurons",
                tolerance=0.01,
            )

    def test_combine_invalid_comparison_type(self):
        for comparison_type in [
            "one-tailed",  # misspelled, dash instead of underscore
            "one tailed",  # misspelled, space instead of underscore
            "three_tailed",  # invalid value
        ]:
            self.assertRaisesRegex(
                IdeasError,
                "The comparison type must be 'two_tailed', "
                "'one_tailed_less', or 'one_tailed_greater'.",
                combine_compare_peri_event_data,
                group1_traces_files=[
                    "group1_traces.csv",
                    "group1B_traces.csv",
                ],
                group1_stats_files=["group1_stats.csv", "group1B_stats.csv"],
                group1_name="Group 1",
                group2_traces_files=[],
                group2_stats_files=[],
                group2_name="",
                comparison_type=comparison_type,
                data_pairing="unpaired",
                significance_threshold=0.05,
                average_method="neurons",
                tolerance=0.01,
            )

    def test_combine_and_compare_invalid_comparison_type(self):
        for comparison_type in [
            "one-tailed",  # misspelled, dash instead of underscore
            "one tailed",  # misspelled, space instead of underscore
            "three_tailed",  # invalid value
            "",  # empty string
        ]:
            self.assertRaisesRegex(
                IdeasError,
                "The comparison type must be 'two_tailed', "
                "'one_tailed_less', or 'one_tailed_greater'.",
                combine_compare_peri_event_data,
                group1_traces_files=[
                    "group1_traces.csv",
                    "group1B_traces.csv",
                ],
                group1_stats_files=["group1_stats.csv", "group1B_stats.csv"],
                group1_name="Group 1",
                group2_traces_files=[
                    "group2_traces.csv",
                    "group2B_traces.csv",
                ],
                group2_stats_files=["group2_stats.csv", "group2B_stats.csv"],
                group2_name="Group 2",
                comparison_type=comparison_type,
                data_pairing="unpaired",
                significance_threshold=0.05,
                average_method="neurons",
                tolerance=0.01,
            )

    def test_combine_invalid_data_pairing(self):
        for data_pairing in [
            "unpaird",  # misspelled, missing 'e'
            "semi-paired",  # invalid value
        ]:
            self.assertRaisesRegex(
                IdeasError,
                "The data pairing parameter must be set to 'paired' or 'unpaired'.",
                combine_compare_peri_event_data,
                group1_traces_files=[
                    "group1A_traces.csv",
                    "group1B_traces.csv",
                ],
                group1_stats_files=["group1A_stats.csv", "group1B_stats.csv"],
                group1_name="Group 1",
                group2_traces_files=[],
                group2_stats_files=[],
                group2_name="",
                comparison_type="two_tailed",
                data_pairing=data_pairing,
                significance_threshold=0.05,
                average_method="neurons",
                tolerance=0.01,
            )

    def test_combine_and_compare_invalid_data_pairing(self):
        for data_pairing in [
            "unpaird",  # misspelled, missing 'e'
            "semi-paired",  # invalid value
            "",  # empty string
        ]:
            self.assertRaisesRegex(
                IdeasError,
                "The data pairing parameter must be set to 'paired' or 'unpaired'.",
                combine_compare_peri_event_data,
                group1_traces_files=[
                    "group1A_traces.csv",
                    "group1B_traces.csv",
                ],
                group1_stats_files=["group1A_stats.csv", "group1B_stats.csv"],
                group1_name="Group 1",
                group2_traces_files=[
                    "group2A_traces.csv",
                    "group2B_traces.csv",
                ],
                group2_stats_files=["group2A_stats.csv", "group2B_stats.csv"],
                group2_name="Group 2",
                comparison_type="two_tailed",
                data_pairing=data_pairing,
                significance_threshold=0.05,
                average_method="neurons",
                tolerance=0.01,
            )

    def test_combine_compare_insufficient_data_for_combining_group1(self):
        # providing just 1 peri-event analysis for group 1 and nothing else
        # means the tool has nothing to combine or compare
        self.assertRaisesRegex(
            IdeasError,
            "The first input group must contain at least two peri-event analyses.",
            combine_compare_peri_event_data,
            group1_traces_files=["group1_traces.csv"],
            group1_stats_files=["group1_stats.csv"],
            group1_name="Group 1",
            group2_traces_files=[],
            group2_stats_files=[],
            group2_name="Group 2",
            comparison_type="two_tailed",
            data_pairing="paired",
            significance_threshold=0.05,
            average_method="neurons",
            tolerance=0.01,
        )

    def test_combine_compare_shifted_time_windows_within_tolerance(self):
        input_dir = os.path.join(self.input_dir, "shifted_time_windows")
        group1_traces_files = [
            os.path.join(input_dir, "group1_subject1.TRACES.csv"),
            os.path.join(input_dir, "group1_subject2.TRACES.csv"),
        ]
        group1_stats_files = [
            os.path.join(input_dir, "group1_subject1.STATISTICS.csv"),
            os.path.join(input_dir, "group1_subject2.STATISTICS.csv"),
        ]
        group2_traces_files = [
            os.path.join(input_dir, "group2_subject1.TRACES.csv"),
            os.path.join(input_dir, "group2_subject2.TRACES.csv"),
        ]
        group2_stats_files = [
            os.path.join(input_dir, "group2_subject1.STATISTICS.csv"),
            os.path.join(input_dir, "group2_subject2.STATISTICS.csv"),
        ]

        group1_name = "wt"
        group2_name = "PER2"

        combine_compare_peri_event_data(
            group1_traces_files=group1_traces_files,
            group1_stats_files=group1_stats_files,
            group1_name=group1_name,
            group2_traces_files=group2_traces_files,
            group2_stats_files=group2_stats_files,
            group2_name=group2_name,
            comparison_type="one_tailed_less",
            data_pairing="unpaired",
            significance_threshold=0.05,
            average_method="neurons",
            tolerance=0.01,
            output_dir=self.output_dir,
        )

        # validate existence of expected output files
        actual_files = os.listdir(self.output_dir)
        expected_files = [
            # output files expected by the system
            # "output_manifest.json",
            "output_metadata.json",
            # output files from combining data in group 1
            f"event_aligned_activity_{group1_name}.csv",
            f"event_aligned_activity_by_modulation_{group1_name}.preview.svg",
            f"fraction_of_modulated_neurons_{group1_name}.preview.svg",
            f"event_aligned_statistics_{group1_name}.csv",
            f"event_aligned_population_activity_{group1_name}.preview.svg",
            f"event_aligned_single_cell_activity_heatmap_{group1_name}.preview.svg",
            # output files from combining data in group 2
            f"event_aligned_activity_{group2_name}.csv",
            f"event_aligned_activity_by_modulation_{group2_name}.preview.svg",
            f"fraction_of_modulated_neurons_{group2_name}.preview.svg",
            f"event_aligned_statistics_{group2_name}.csv",
            f"event_aligned_population_activity_{group2_name}.preview.svg",
            f"event_aligned_single_cell_activity_heatmap_{group2_name}.preview.svg",
            # comparison output files
            "comparison_data.csv",
            "event_aligned_population_activity_comparison.preview.svg",
            "event_aligned_subpopulation_activity_comparison.preview.svg",
            "modulated_fractions_comparison.preview.svg",
        ]
        self.assertEqual(sorted(actual_files), sorted(expected_files))

        # validate output manifest
        # expected_output_manifest = {
        #     "schema_version": "2.0.0",
        #     "groups": [
        #         {
        #             "group_key": "combine_compare_peri_event_data_output",
        #             "group_type": "tool_output",
        #             "group_id": "3e1896a3-cbe8-408d-9f49-73bca135c1d8",
        #             "series": [],
        #             "files": [
        #                 {
        #                     "file_key": "group1_traces_files",
        #                     "file_name": "group1_subject1.TRACES.csv",
        #                     "file_id": "d295b776-c43a-41bb-b2dc-108f8ed0b393",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_"
        #                     "event_data/shifted_time_windows/group1_subject1"
        #                     ".TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_traces_files",
        #                     "file_name": "group1_subject2.TRACES.csv",
        #                     "file_id": "71da8241-6f42-46a8-8905-8da60057a148",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event"
        #                     "_data/shifted_time_windows/group1_subject2.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_stats_files",
        #                     "file_name": "group1_subject1.STATISTICS.csv",
        #                     "file_id": "50e5f8d0-d1dc-481b-ae92-4f96afcbe78b",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event"
        #                     "_data/shifted_time_windows/group1_subject1.STATISTICS"
        #                     ".csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_stats_files",
        #                     "file_name": "group1_subject2.STATISTICS.csv",
        #                     "file_id": "2cbc23d0-8237-47d3-9fbe-9ec558f64c8e",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event"
        #                     "_data/shifted_time_windows/group1_subject2.STATISTICS"
        #                     ".csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group2_traces_files",
        #                     "file_name": "group2_subject1.TRACES.csv",
        #                     "file_id": "cb8e1c16-1982-4f9e-a4db-e9901e2485ba",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event"
        #                     "_data/shifted_time_windows/group2_subject1.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group2_traces_files",
        #                     "file_name": "group2_subject2.TRACES.csv",
        #                     "file_id": "de791c85-b7e3-4784-b7e8-0dfb7efaffd1",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event"
        #                     "_data/shifted_time_windows/group2_subject2.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group2_stats_files",
        #                     "file_name": "group2_subject1.STATISTICS.csv",
        #                     "file_id": "2c07158c-c55b-4325-82aa-a40e53333e78",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event"
        #                     "_data/shifted_time_windows/group2_subject1.STATISTICS"
        #                     ".csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group2_stats_files",
        #                     "file_name": "group2_subject2.STATISTICS.csv",
        #                     "file_id": "829ce922-d4ac-4549-a5f0-4c5ced9e9630",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare_peri_event_data"
        #                     "/shifted_time_windows/group2_subject2.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_event_aligned_activity_traces_event_"
        #                     "aligned_activity_traces_csv_file",
        #                     "file_name": "event_aligned_activity_wt.csv",
        #                     "file_id": "7b02655f-a670-4232-8685-61d3dd4c084a",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data/event_"
        #                     "aligned_activity_wt.csv",
        #                     "file_type": "combined_peri_event_traces",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned population activity figure",
        #                             "help": "Event-aligned average population activity line plot",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "event_aligned_population_activity_wt.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "event_aligned_single_cell_activity_heatmap_"
        #                             "wt.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group1_event_aligned_statistics_event_aligned_"
        #                     "activity_statistics_csv_file",
        #                     "file_name": "event_aligned_statistics_wt.csv",
        #                     "file_id": "b9f5e661-9dda-4c0d-89f4-a7899ce24a90",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data/event_"
        #                     "aligned_statistics_wt.csv",
        #                     "file_type": "combined_peri_event_statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity "
        #                             "line plot (up-, down-, and non-modulated neurons)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/event"
        #                             "_aligned_activity_by_modulation_wt.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in each "
        #                             "sub-population (up-, down-, and non-modulated neurons)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/fraction"
        #                             "_of_modulated_neurons_wt.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group2_event_aligned_activity_traces_event_aligned_"
        #                     "activity_traces_csv_file",
        #                     "file_name": "event_aligned_activity_PER2.csv",
        #                     "file_id": "4775eb9e-15f9-4ba2-aeca-ad1df7944328",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data/event_"
        #                     "aligned_activity_PER2.csv",
        #                     "file_type": "combined_peri_event_traces",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned population activity figure",
        #                             "help": "Event-aligned average population activity line plot",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "event_aligned_population_activity_PER2.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "event_aligned_single_cell_activity_heatmap"
        #                             "_PER2.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group2_event_aligned_statistics_event_aligned_"
        #                     "activity_statistics_csv_file",
        #                     "file_name": "event_aligned_statistics_PER2.csv",
        #                     "file_id": "3b8c2926-1d98-425b-a10d-e7660ae3d0f4",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data/event"
        #                     "_aligned_statistics_PER2.csv",
        #                     "file_type": "combined_peri_event_statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity "
        #                             "line plot (up-, down-, and non-modulated neurons)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "event_aligned_activity_by_modulation_PER2.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in "
        #                             "each sub-population (up-, down-, and non-modulated neurons)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "fraction_of_modulated_neurons_PER2.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "comparison_data_comparison_data_csv_file",
        #                     "file_name": "comparison_data.csv",
        #                     "file_id": "6d4ad30a-83b9-4032-b9c3-16d53b9015cd",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                     "comparison_data.csv",
        #                     "file_type": "peri_event_comparison_data",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned population activity figure",
        #                             "help": "Event-aligned average population activity "
        #                             "line plot",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "event_aligned_population_activity_comparison"
        #                             ".preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity "
        #                             "line plot",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "event_aligned_subpopulation_activity_comparison"
        #                             ".preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each modulation group",
        #                             "help": "Fraction of neurons in each modulation group",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "modulated_fractions_comparison.preview.svg",
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
        #         expected_output_manifest=expected_output_manifest,
        #         tool_metadata=self.tool_metadata[self.tool_key],
        #         actual_manifest_data_filepath=self.output_manifest_file,
        #     )
        # )

    def test_combine_compare_shifted_time_windows_exceeds_tolerance(self):
        input_dir = os.path.join(self.input_dir, "shifted_time_windows")
        group1_traces_files = [
            os.path.join(input_dir, "group1_subject1.TRACES.csv"),
            os.path.join(input_dir, "group1_subject2.TRACES.csv"),
        ]
        group1_stats_files = [
            os.path.join(input_dir, "group1_subject1.STATISTICS.csv"),
            os.path.join(input_dir, "group1_subject2.STATISTICS.csv"),
        ]
        group2_traces_files = [
            os.path.join(input_dir, "group2_subject1.TRACES.csv"),
            os.path.join(input_dir, "group2_subject2.TRACES.csv"),
        ]
        group2_stats_files = [
            os.path.join(input_dir, "group2_subject1.STATISTICS.csv"),
            os.path.join(input_dir, "group2_subject2.STATISTICS.csv"),
        ]

        self.assertRaisesRegex(
            IdeasError,
            "The difference in the time windows of the input traces files "
            "'group2_subject2.TRACES.csv' and 'group2_subject1.TRACES.csv' "
            "exceeds the specified tolerance.",
            combine_compare_peri_event_data,
            group1_traces_files=group1_traces_files,
            group1_stats_files=group1_stats_files,
            group1_name="wt",
            group2_traces_files=group2_traces_files,
            group2_stats_files=group2_stats_files,
            group2_name="PER2",
            comparison_type="one_tailed_greater",
            data_pairing="unpaired",
            significance_threshold=0.05,
            average_method="neurons",
            tolerance=0.001,
            output_dir=self.output_dir,
        )

    def test_combine_peri_event_with_shuffled_data_single_group(self):
        # ALL peri-event traces files contain population shuffled data
        group1_traces_files = [
            os.path.join(self.input_dir, f)
            for f in [
                "group1/group1_subject1_with_shuffled_data.TRACES.csv",
                "group1/group1_subject2_with_shuffled_data.TRACES.csv",
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
        combine_compare_peri_event_data(
            group1_traces_files=group1_traces_files,
            group1_stats_files=group1_stats_files,
            group1_name=group_name,
            group2_traces_files=[],
            group2_stats_files=[],
            group2_name="",
            comparison_type=None,
            data_pairing=None,
            significance_threshold=None,
            average_method="neurons",
            tolerance=0.01,
            output_dir=self.output_dir,
        )

        # validate existence of expected output files
        actual_files = os.listdir(self.output_dir)
        expected_files = [
            # "output_manifest.json",
            "output_metadata.json",
            f"event_aligned_activity_{group_name}.csv",
            f"event_aligned_activity_by_modulation_{group_name}.preview.svg",
            f"fraction_of_modulated_neurons_{group_name}.preview.svg",
            f"event_aligned_statistics_{group_name}.csv",
            f"event_aligned_population_activity_{group_name}.preview.svg",
            f"event_aligned_single_cell_activity_heatmap_{group_name}.preview.svg",
        ]
        self.assertEqual(sorted(actual_files), sorted(expected_files))

        # validate event-aligned traces
        df_traces = pd.read_csv(
            os.path.join(
                self.output_dir, f"event_aligned_activity_{group_name}.csv"
            )
        )
        self.assertEqual(df_traces.shape, (119, 249))

        # validate population mean and sem (the expected values were computed manually)
        self.assertAlmostEqual(
            -0.00139902927655282, df_traces["population_mean"][0]
        )
        self.assertAlmostEqual(
            0.00661793085084028, df_traces["population_sem"][0]
        )

        # validate that population shuffled data is not included in the combined data file
        self.assertTrue("population_shuffled_mean" not in df_traces.columns)

        # validate sub-population mean and sem
        self.assertAlmostEqual(
            -0.049593321516071064, df_traces["up_modulated_mean"][0]
        )
        self.assertAlmostEqual(
            0.02248465911945461, df_traces["up_modulated_sem"][0]
        )
        self.assertAlmostEqual(
            0.0284092191640714, df_traces["down_modulated_mean"][0]
        )
        self.assertAlmostEqual(
            0.0325825083730468, df_traces["down_modulated_sem"][0]
        )
        self.assertAlmostEqual(
            0.000372933333441427, df_traces["non_modulated_mean"][0]
        )
        self.assertAlmostEqual(
            0.006994805756296599, df_traces["non_modulated_sem"][0]
        )

        # validate event-aligned statistics
        df_stats = pd.read_csv(
            os.path.join(
                self.output_dir, f"event_aligned_statistics_{group_name}.csv"
            )
        )
        self.assertEqual(df_stats.shape, (240, 8))

        expected_cols = [
            "name",
            "true_mean_post-pre",
            "shuffled_mean_post-pre",
            "shuffled_std_post-pre",
            "z-score",
            "p-value",
            "modulation",
            "file",
        ]
        actual_cols = df_stats.columns.tolist()
        self.assertEqual(actual_cols, expected_cols)

        # ensure the number of modulated cells match the input
        # since we used the same p-value to reclassify them (p=0.05)
        self.assertEqual(
            7,
            len(
                df_stats[
                    (df_stats.modulation == 1.0)
                    & (df_stats.file == "group1_subject1.STATISTICS.csv")
                ]
            ),
        )
        self.assertEqual(
            6,
            len(
                df_stats[
                    (df_stats.modulation == -1.0)
                    & (df_stats.file == "group1_subject1.STATISTICS.csv")
                ]
            ),
        )
        self.assertEqual(
            107,
            len(
                df_stats[
                    (df_stats.modulation == 0.0)
                    & (df_stats.file == "group1_subject1.STATISTICS.csv")
                ]
            ),
        )

        self.assertEqual(
            6,
            len(
                df_stats[
                    (df_stats.modulation == 1.0)
                    & (df_stats.file == "group1_subject2.STATISTICS.csv")
                ]
            ),
        )
        self.assertEqual(
            2,
            len(
                df_stats[
                    (df_stats.modulation == -1.0)
                    & (df_stats.file == "group1_subject2.STATISTICS.csv")
                ]
            ),
        )
        self.assertEqual(
            112,
            len(
                df_stats[
                    (df_stats.modulation == 0.0)
                    & (df_stats.file == "group1_subject2.STATISTICS.csv")
                ]
            ),
        )

        # # validate output manifest
        # expected_output_manifest = {
        #     "schema_version": "2.0.0",
        #     "groups": [
        #         {
        #             "group_key": "combine_compare_peri_event_data_output",
        #             "group_type": "tool_output",
        #             "group_id": "ba34b173-353d-45b4-9dfa-8e4ddab00090",
        #             "series": [],
        #             "files": [
        #                 {
        #                     "file_key": "group1_traces_files",
        #                     "file_name": "group1_subject1_with_shuffled_data.TRACES.csv",
        #                     "file_id": "28b5fa10-f3c3-43cd-87b2-da6b4f3fbbfa",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare"
        #                     "_peri_event_data/group1/group1_subject1_with_shuffled_data"
        #                     ".TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_traces_files",
        #                     "file_name": "group1_subject2_with_shuffled_data.TRACES.csv",
        #                     "file_id": "4252298c-eb35-45bf-91d9-58fba8841501",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare"
        #                     "_peri_event_data/group1/group1_subject2_with_shuffled_data"
        #                     ".TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_stats_files",
        #                     "file_name": "group1_subject1.STATISTICS.csv",
        #                     "file_id": "240a79cf-7872-4fd7-a687-7710667adb9a",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare"
        #                     "_peri_event_data/group1/group1_subject1"
        #                     ".STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_stats_files",
        #                     "file_name": "group1_subject2.STATISTICS.csv",
        #                     "file_id": "578d0ee6-525d-4a87-b9f3-96a044938977",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_"
        #                     "compare_peri_event_data/group1/group1"
        #                     "_subject2.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_event_aligned_activity_"
        #                     "traces_event_aligned_activity_traces"
        #                     "_csv_file",
        #                     "file_name": "event_aligned_activity_group1.csv",
        #                     "file_id": "4150ca1f-90e0-469f-ba71-d4b509d40560",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                     "event_aligned_activity_group1.csv",
        #                     "file_type": "combined_peri_event_traces",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned population activity figure",
        #                             "help": "Event-aligned average population "
        #                             "activity line plot",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "event_aligned_population_activity_group1"
        #                             ".preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "event_aligned_single_cell_activity_heatmap"
        #                             "_group1.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group1_event_aligned_statistics_event_"
        #                     "aligned_activity_statistics_csv_file",
        #                     "file_name": "event_aligned_statistics_group1.csv",
        #                     "file_id": "76a0a431-3d02-45ca-a21a-64244e9ed5c9",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                     "event_aligned_statistics_group1.csv",
        #                     "file_type": "combined_peri_event_statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity "
        #                             "line plot (up-, down-, and non-modulated neurons)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "event_aligned_activity_by_modulation_group1"
        #                             ".preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in"
        #                             " each sub-population (up-, down-, and non-modulated "
        #                             "neurons)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "fraction_of_modulated_neurons_group1"
        #                             ".preview.svg",
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
        #         expected_output_manifest=expected_output_manifest,
        #         tool_metadata=self.tool_metadata[self.tool_key],
        #         actual_manifest_data_filepath=self.output_manifest_file,
        #     )
        # )

    def test_combine_peri_event_with_mixed_data_single_group(self):
        # SOME but not all peri-event traces files contain population shuffled data
        group1_traces_files = [
            os.path.join(self.input_dir, f)
            for f in [
                "group1/group1_subject1.TRACES.csv",
                "group1/group1_subject2_with_shuffled_data.TRACES.csv",
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
        combine_compare_peri_event_data(
            group1_traces_files=group1_traces_files,
            group1_stats_files=group1_stats_files,
            group1_name=group_name,
            group2_traces_files=[],
            group2_stats_files=[],
            group2_name="",
            comparison_type=None,
            data_pairing=None,
            significance_threshold=None,
            average_method="neurons",
            tolerance=0.01,
            output_dir=self.output_dir,
        )

        # validate existence of expected output files
        actual_files = os.listdir(self.output_dir)
        expected_files = [
            # "output_manifest.json",
            "output_metadata.json",
            f"event_aligned_activity_{group_name}.csv",
            f"event_aligned_activity_by_modulation_{group_name}.preview.svg",
            f"fraction_of_modulated_neurons_{group_name}.preview.svg",
            f"event_aligned_statistics_{group_name}.csv",
            f"event_aligned_population_activity_{group_name}.preview.svg",
            f"event_aligned_single_cell_activity_heatmap_{group_name}.preview.svg",
        ]
        self.assertEqual(sorted(actual_files), sorted(expected_files))

        # validate event-aligned traces
        df_traces = pd.read_csv(
            os.path.join(
                self.output_dir, f"event_aligned_activity_{group_name}.csv"
            )
        )
        self.assertEqual(df_traces.shape, (119, 249))

        # validate population mean and sem (the expected values were computed manually)
        self.assertAlmostEqual(
            -0.00139902927655282, df_traces["population_mean"][0]
        )
        self.assertAlmostEqual(
            0.00661793085084028, df_traces["population_sem"][0]
        )

        # validate that population shuffled data is not included in the combined data file
        self.assertTrue("population_shuffled_mean" not in df_traces.columns)

        # validate sub-population mean and sem
        self.assertAlmostEqual(
            -0.049593321516071064, df_traces["up_modulated_mean"][0]
        )
        self.assertAlmostEqual(
            0.02248465911945461, df_traces["up_modulated_sem"][0]
        )
        self.assertAlmostEqual(
            0.0284092191640714, df_traces["down_modulated_mean"][0]
        )
        self.assertAlmostEqual(
            0.0325825083730468, df_traces["down_modulated_sem"][0]
        )
        self.assertAlmostEqual(
            0.000372933333441427, df_traces["non_modulated_mean"][0]
        )
        self.assertAlmostEqual(
            0.006994805756296599, df_traces["non_modulated_sem"][0]
        )

        # validate event-aligned statistics
        df_stats = pd.read_csv(
            os.path.join(
                self.output_dir, f"event_aligned_statistics_{group_name}.csv"
            )
        )
        self.assertEqual(df_stats.shape, (240, 8))

        expected_cols = [
            "name",
            "true_mean_post-pre",
            "shuffled_mean_post-pre",
            "shuffled_std_post-pre",
            "z-score",
            "p-value",
            "modulation",
            "file",
        ]
        actual_cols = df_stats.columns.tolist()
        self.assertEqual(actual_cols, expected_cols)

        # ensure the number of modulated cells match the input
        # since we used the same p-value to reclassify them (p=0.05)
        self.assertEqual(
            7,
            len(
                df_stats[
                    (df_stats.modulation == 1.0)
                    & (df_stats.file == "group1_subject1.STATISTICS.csv")
                ]
            ),
        )
        self.assertEqual(
            6,
            len(
                df_stats[
                    (df_stats.modulation == -1.0)
                    & (df_stats.file == "group1_subject1.STATISTICS.csv")
                ]
            ),
        )
        self.assertEqual(
            107,
            len(
                df_stats[
                    (df_stats.modulation == 0.0)
                    & (df_stats.file == "group1_subject1.STATISTICS.csv")
                ]
            ),
        )

        self.assertEqual(
            6,
            len(
                df_stats[
                    (df_stats.modulation == 1.0)
                    & (df_stats.file == "group1_subject2.STATISTICS.csv")
                ]
            ),
        )
        self.assertEqual(
            2,
            len(
                df_stats[
                    (df_stats.modulation == -1.0)
                    & (df_stats.file == "group1_subject2.STATISTICS.csv")
                ]
            ),
        )
        self.assertEqual(
            112,
            len(
                df_stats[
                    (df_stats.modulation == 0.0)
                    & (df_stats.file == "group1_subject2.STATISTICS.csv")
                ]
            ),
        )

        # validate output manifest
        # expected_output_manifest = {
        #     "schema_version": "2.0.0",
        #     "groups": [
        #         {
        #             "group_key": "combine_compare_peri_event_data_output",
        #             "group_type": "tool_output",
        #             "group_id": "ba34b173-353d-45b4-9dfa-8e4ddab00090",
        #             "series": [],
        #             "files": [
        #                 {
        #                     "file_key": "group1_traces_files",
        #                     "file_name": "group1_subject1.TRACES.csv",
        #                     "file_id": "28b5fa10-f3c3-43cd-87b2-da6b4f3fbbfa",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare"
        #                     "_peri_event_data/group1/group1_subject1"
        #                     ".TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_traces_files",
        #                     "file_name": "group1_subject2_with_shuffled_data.TRACES.csv",
        #                     "file_id": "4252298c-eb35-45bf-91d9-58fba8841501",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare"
        #                     "_peri_event_data/group1/group1_subject2_with_shuffled_data"
        #                     ".TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_stats_files",
        #                     "file_name": "group1_subject1.STATISTICS.csv",
        #                     "file_id": "240a79cf-7872-4fd7-a687-7710667adb9a",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_compare"
        #                     "_peri_event_data/group1/group1_subject1"
        #                     ".STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_stats_files",
        #                     "file_name": "group1_subject2.STATISTICS.csv",
        #                     "file_id": "578d0ee6-525d-4a87-b9f3-96a044938977",
        #                     "file_path": "/ideas/toolbox/tests/data/combine_"
        #                     "compare_peri_event_data/group1/group1"
        #                     "_subject2.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_event_aligned_activity_"
        #                     "traces_event_aligned_activity_traces"
        #                     "_csv_file",
        #                     "file_name": "event_aligned_activity_group1.csv",
        #                     "file_id": "4150ca1f-90e0-469f-ba71-d4b509d40560",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                     "event_aligned_activity_group1.csv",
        #                     "file_type": "combined_peri_event_traces",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned population activity figure",
        #                             "help": "Event-aligned average population "
        #                             "activity line plot",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "event_aligned_population_activity_group1"
        #                             ".preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "event_aligned_single_cell_activity_heatmap"
        #                             "_group1.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group1_event_aligned_statistics_event_"
        #                     "aligned_activity_statistics_csv_file",
        #                     "file_name": "event_aligned_statistics_group1.csv",
        #                     "file_id": "76a0a431-3d02-45ca-a21a-64244e9ed5c9",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                     "event_aligned_statistics_group1.csv",
        #                     "file_type": "combined_peri_event_statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity "
        #                             "line plot (up-, down-, and non-modulated neurons)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "event_aligned_activity_by_modulation_group1"
        #                             ".preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in"
        #                             " each sub-population (up-, down-, and non-modulated "
        #                             "neurons)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "fraction_of_modulated_neurons_group1"
        #                             ".preview.svg",
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
        #         expected_output_manifest=expected_output_manifest,
        #         tool_metadata=self.tool_metadata[self.tool_key],
        #         actual_manifest_data_filepath=self.output_manifest_file,
        #     )
        # )

    def test_combine_peri_event_with_shuffled_data_two_groups(self):
        group1_name = "group1"
        group1_traces_files = [
            os.path.join(self.input_dir, f)
            for f in [
                "group1/group1_subject1_with_shuffled_data.TRACES.csv",
                "group1/group1_subject2_with_shuffled_data.TRACES.csv",
            ]
        ]
        group1_stats_files = [
            os.path.join(self.input_dir, f)
            for f in [
                "group1/group1_subject1.STATISTICS.csv",
                "group1/group1_subject2.STATISTICS.csv",
            ]
        ]

        group2_name = "group2"
        group2_traces_files = [
            os.path.join(self.input_dir, f)
            for f in [
                "group2/group2_subject1_with_shuffled_data.TRACES.csv",
                "group2/group2_subject2_with_shuffled_data.TRACES.csv",
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
        combine_compare_peri_event_data(
            group1_traces_files=group1_traces_files,
            group1_stats_files=group1_stats_files,
            group1_name=group1_name,
            group2_traces_files=group2_traces_files,
            group2_stats_files=group2_stats_files,
            group2_name=group2_name,
            comparison_type="two_tailed",
            data_pairing="paired",
            significance_threshold=None,
            average_method="neurons",
            tolerance=0.01,
            output_dir=self.output_dir,
        )

        # validate existence of expected output files
        actual_files = os.listdir(self.output_dir)
        expected_files = [
            # output files expected by the system
            # "output_manifest.json",
            "output_metadata.json",
            # output files from combining data in group 1
            f"event_aligned_activity_{group1_name}.csv",
            f"event_aligned_activity_by_modulation_{group1_name}.preview.svg",
            f"fraction_of_modulated_neurons_{group1_name}.preview.svg",
            f"event_aligned_statistics_{group1_name}.csv",
            f"event_aligned_population_activity_{group1_name}.preview.svg",
            f"event_aligned_single_cell_activity_heatmap_{group1_name}.preview.svg",
            # output files from combining data in group 2
            f"event_aligned_activity_{group2_name}.csv",
            f"event_aligned_activity_by_modulation_{group2_name}.preview.svg",
            f"fraction_of_modulated_neurons_{group2_name}.preview.svg",
            f"event_aligned_statistics_{group2_name}.csv",
            f"event_aligned_population_activity_{group2_name}.preview.svg",
            f"event_aligned_single_cell_activity_heatmap_{group2_name}.preview.svg",
            # comparison output files
            "comparison_data.csv",
            "event_aligned_population_activity_comparison.preview.svg",
            "event_aligned_subpopulation_activity_comparison.preview.svg",
            "modulated_fractions_comparison.preview.svg",
        ]
        self.assertEqual(sorted(actual_files), sorted(expected_files))

        # validate event-aligned traces for GROUP 1
        df_traces_group1 = pd.read_csv(
            os.path.join(
                self.output_dir, f"event_aligned_activity_{group1_name}.csv"
            )
        )
        self.assertEqual(df_traces_group1.shape, (119, 249))

        # validate population mean and sem (the expected values were computed manually)
        self.assertAlmostEqual(
            -0.00139902927655282, df_traces_group1["population_mean"][0]
        )
        self.assertAlmostEqual(
            0.00661793085084028, df_traces_group1["population_sem"][0]
        )

        # validate that no shuffled data is present in the csv file
        self.assertTrue(
            "population_shuffled_mean" not in df_traces_group1.columns
        )

        # validate sub-population mean and sem
        self.assertAlmostEqual(
            -0.049593321516071064, df_traces_group1["up_modulated_mean"][0]
        )
        self.assertAlmostEqual(
            0.0284092191640714, df_traces_group1["down_modulated_mean"][0]
        )
        self.assertAlmostEqual(
            0.000372933333441427, df_traces_group1["non_modulated_mean"][0]
        )

        # validate event-aligned statistics for GROUP 1
        df_stats_group1 = pd.read_csv(
            os.path.join(
                self.output_dir, f"event_aligned_statistics_{group1_name}.csv"
            )
        )
        self.assertEqual(df_stats_group1.shape, (240, 8))

        expected_cols = [
            "name",
            "true_mean_post-pre",
            "shuffled_mean_post-pre",
            "shuffled_std_post-pre",
            "z-score",
            "p-value",
            "modulation",
            "file",
        ]
        actual_cols = df_stats_group1.columns.tolist()
        self.assertEqual(actual_cols, expected_cols)

        # ensure the number of modulated cells match the input
        # since we used the same p-value to reclassify them (p=0.05)
        self.assertEqual(
            7,
            len(
                df_stats_group1[
                    (df_stats_group1.modulation == 1.0)
                    & (
                        df_stats_group1.file
                        == "group1_subject1.STATISTICS.csv"
                    )
                ]
            ),
        )
        self.assertEqual(
            6,
            len(
                df_stats_group1[
                    (df_stats_group1.modulation == -1.0)
                    & (
                        df_stats_group1.file
                        == "group1_subject1.STATISTICS.csv"
                    )
                ]
            ),
        )
        self.assertEqual(
            107,
            len(
                df_stats_group1[
                    (df_stats_group1.modulation == 0.0)
                    & (
                        df_stats_group1.file
                        == "group1_subject1.STATISTICS.csv"
                    )
                ]
            ),
        )

        self.assertEqual(
            6,
            len(
                df_stats_group1[
                    (df_stats_group1.modulation == 1.0)
                    & (
                        df_stats_group1.file
                        == "group1_subject2.STATISTICS.csv"
                    )
                ]
            ),
        )
        self.assertEqual(
            2,
            len(
                df_stats_group1[
                    (df_stats_group1.modulation == -1.0)
                    & (
                        df_stats_group1.file
                        == "group1_subject2.STATISTICS.csv"
                    )
                ]
            ),
        )
        self.assertEqual(
            112,
            len(
                df_stats_group1[
                    (df_stats_group1.modulation == 0.0)
                    & (
                        df_stats_group1.file
                        == "group1_subject2.STATISTICS.csv"
                    )
                ]
            ),
        )

        # validate event-aligned traces for GROUP 2
        df_traces_group2 = pd.read_csv(
            os.path.join(
                self.output_dir, f"event_aligned_activity_{group2_name}.csv"
            )
        )
        self.assertEqual(df_traces_group2.shape, (119, 249))

        # validate population mean and sem (the expected values were computed manually)
        self.assertAlmostEqual(
            -0.0278609511299807, df_traces_group2["population_mean"][0]
        )
        self.assertAlmostEqual(
            0.00694676565981122, df_traces_group2["population_sem"][0]
        )

        # validate sub-population mean and sem
        self.assertAlmostEqual(
            -0.10627978328120656, df_traces_group2["up_modulated_mean"][0]
        )
        self.assertAlmostEqual(
            0.09017987368724932, df_traces_group2["down_modulated_mean"][0]
        )
        self.assertAlmostEqual(
            -0.02652140411247937, df_traces_group2["non_modulated_mean"][0]
        )

        # validate event-aligned statistics for GROUP 2
        df_stats_group2 = pd.read_csv(
            os.path.join(
                self.output_dir, f"event_aligned_statistics_{group2_name}.csv"
            )
        )
        self.assertEqual(df_stats_group2.shape, (240, 8))

        expected_cols = [
            "name",
            "true_mean_post-pre",
            "shuffled_mean_post-pre",
            "shuffled_std_post-pre",
            "z-score",
            "p-value",
            "modulation",
            "file",
        ]
        actual_cols = df_stats_group2.columns.tolist()
        self.assertEqual(actual_cols, expected_cols)

        # ensure the number of modulated cells match the input
        # since we used the same p-value to reclassify them (p=0.05)
        self.assertEqual(
            23,
            len(
                df_stats_group2[
                    (df_stats_group2.modulation == 1.0)
                    & (
                        df_stats_group2.file
                        == "group2_subject1.STATISTICS.csv"
                    )
                ]
            ),
        )
        self.assertEqual(
            18,
            len(
                df_stats_group2[
                    (df_stats_group2.modulation == -1.0)
                    & (
                        df_stats_group2.file
                        == "group2_subject1.STATISTICS.csv"
                    )
                ]
            ),
        )
        self.assertEqual(
            79,
            len(
                df_stats_group2[
                    (df_stats_group2.modulation == 0.0)
                    & (
                        df_stats_group2.file
                        == "group2_subject1.STATISTICS.csv"
                    )
                ]
            ),
        )

        self.assertEqual(
            22,
            len(
                df_stats_group2[
                    (df_stats_group2.modulation == 1.0)
                    & (
                        df_stats_group2.file
                        == "group2_subject2.STATISTICS.csv"
                    )
                ]
            ),
        )
        self.assertEqual(
            10,
            len(
                df_stats_group2[
                    (df_stats_group2.modulation == -1.0)
                    & (
                        df_stats_group2.file
                        == "group2_subject2.STATISTICS.csv"
                    )
                ]
            ),
        )
        self.assertEqual(
            88,
            len(
                df_stats_group2[
                    (df_stats_group2.modulation == 0.0)
                    & (
                        df_stats_group2.file
                        == "group2_subject2.STATISTICS.csv"
                    )
                ]
            ),
        )

        # validate output manifest
        # expected_output_manifest = {
        #     "schema_version": "2.0.0",
        #     "groups": [
        #         {
        #             "group_key": "combine_compare_peri_event_data_output",
        #             "group_type": "tool_output",
        #             "group_id": "7765d844-71e1-4203-af4c-3a4134dab37f",
        #             "series": [],
        #             "files": [
        #                 {
        #                     "file_key": "group1_traces_files",
        #                     "file_name": "group1_subject1_with_shuffled_data.TRACES.csv",
        #                     "file_id": "845e3522-91fe-459f-b05a-fafb15a93aa4",
        #                     "file_path": "/ideas/toolbox/tests/data/"
        #                     "combine_compare_peri_event_data/"
        #                     "group1/group1_subject1_with_shuffled_data.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_traces_files",
        #                     "file_name": "group1_subject2_with_shuffled_data.TRACES.csv",
        #                     "file_id": "9ef2da72-5622-4c07-90dd-5f59797ffaf5",
        #                     "file_path": "/ideas/toolbox/tests/data/"
        #                     "combine_compare_peri_event_data/"
        #                     "group1/group1_subject2_with_shuffled_data.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_stats_files",
        #                     "file_name": "group1_subject1.STATISTICS.csv",
        #                     "file_id": "504b21d5-5475-4447-ad13-5499da4834e3",
        #                     "file_path": "/ideas/toolbox/tests/data/"
        #                     "combine_compare_peri_event_data/"
        #                     "group1/group1_subject1.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_stats_files",
        #                     "file_name": "group1_subject2.STATISTICS.csv",
        #                     "file_id": "5a3749b8-8af2-454b-9172-2eccb122ec11",
        #                     "file_path": "/ideas/toolbox/tests/data/"
        #                     "combine_compare_peri_event_data/"
        #                     "group1/group1_subject2.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group2_traces_files",
        #                     "file_name": "group2_subject1_with_shuffled_data.TRACES.csv",
        #                     "file_id": "bc9d47f8-d8e9-452d-8de1-480eca7169aa",
        #                     "file_path": "/ideas/toolbox/tests/data/"
        #                     "combine_compare_peri_event_data/"
        #                     "group2/group2_subject1_with_shuffled_data.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group2_traces_files",
        #                     "file_name": "group2_subject2_with_shuffled_data.TRACES.csv",
        #                     "file_id": "8708f587-9c88-414a-982c-07774bfed02f",
        #                     "file_path": "/ideas/toolbox/tests/data/"
        #                     "combine_compare_peri_event_data/"
        #                     "group2/group2_subject2_with_shuffled_data.TRACES.csv",
        #                     "file_type": "event_aligned_neural_data",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group2_stats_files",
        #                     "file_name": "group2_subject1.STATISTICS.csv",
        #                     "file_id": "33ee110c-307f-44ad-8173-ad0f43cfac3c",
        #                     "file_path": "/ideas/toolbox/tests/data/"
        #                     "combine_compare_peri_event_data/"
        #                     "group2/group2_subject1.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group2_stats_files",
        #                     "file_name": "group2_subject2.STATISTICS.csv",
        #                     "file_id": "0dbcff3e-b1a4-483a-96b8-509e603c48c5",
        #                     "file_path": "/ideas/toolbox/tests/data/"
        #                     "combine_compare_peri_event_data/"
        #                     "group2/group2_subject2.STATISTICS.csv",
        #                     "file_type": "statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "source",
        #                 },
        #                 {
        #                     "file_key": "group1_event_aligned_activity_traces_"
        #                     "event_aligned_activity_traces_csv_file",
        #                     "file_name": "event_aligned_activity_group1.csv",
        #                     "file_id": "05f93015-d371-47ae-b838-dc29ec44cf7e",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                     "event_aligned_activity_group1.csv",
        #                     "file_type": "combined_peri_event_traces",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned population activity figure",
        #                             "help": "Event-aligned average population activity line plot",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "event_aligned_population_activity"
        #                             "_group1.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "event_aligned_single_cell_activity_heatmap"
        #                             "_group1.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group1_event_aligned_statistics_event_"
        #                     "aligned_activity_statistics_csv_file",
        #                     "file_name": "event_aligned_statistics_group1.csv",
        #                     "file_id": "ed5ad3d9-fe28-492e-8c7b-3ae432fd34a7",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                     "event_aligned_statistics_group1.csv",
        #                     "file_type": "combined_peri_event_statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity "
        #                             "line plot (up-, down-, and non-modulated neurons)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "event_aligned_activity_by_modulation_group1"
        #                             ".preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in "
        #                             "each sub-population (up-, down-, and non-modulated"
        #                             " neurons)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "fraction_of_modulated_neurons_group1.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group2_event_aligned_activity_traces_event"
        #                     "_aligned_activity_traces_csv_file",
        #                     "file_name": "event_aligned_activity_group2.csv",
        #                     "file_id": "23835d2d-56c1-4a54-b5c9-0ff9df0f199b",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                     "event_aligned_activity_group2.csv",
        #                     "file_type": "combined_peri_event_traces",
        #                     "file_format": "csv",
        #                     "file_structure": "time_series",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned population activity figure",
        #                             "help": "Event-aligned average population activity line plot",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "event_aligned_population_activity_group2"
        #                             ".preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned single-cell activity figure",
        #                             "help": "Event-aligned single-cell activity heatmap",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "event_aligned_single_cell_activity_heatmap"
        #                             "_group2.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "group2_event_aligned_statistics_event"
        #                     "_aligned_activity_statistics_csv_file",
        #                     "file_name": "event_aligned_statistics_group2.csv",
        #                     "file_id": "0dba3dc1-0364-4851-b2d8-7fe0a9577a7e",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                     "event_aligned_statistics_group2.csv",
        #                     "file_type": "combined_peri_event_statistics",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity "
        #                             "line plot (up-, down-, and non-modulated neurons)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "event_aligned_activity_by_modulation_group2"
        #                             ".preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each sub-population",
        #                             "help": "Pie chart depicting the fraction of neurons in "
        #                             "each sub-population (up-, down-, and non-modulated"
        #                             " neurons)",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "fraction_of_modulated_neurons_group2.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                     ],
        #                 },
        #                 {
        #                     "file_key": "comparison_data_comparison_data_csv_file",
        #                     "file_name": "comparison_data.csv",
        #                     "file_id": "a6b3f95b-aca6-42db-9f1f-81a6f9adfbd1",
        #                     "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                     "comparison_data.csv",
        #                     "file_type": "peri_event_comparison_data",
        #                     "file_format": "csv",
        #                     "file_structure": "table",
        #                     "file_category": "result",
        #                     "preview": [
        #                         {
        #                             "name": "Event-aligned population activity figure",
        #                             "help": "Event-aligned average population activity line plot",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "event_aligned_population_activity_"
        #                             "comparison.preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Event-aligned sub-population activity figure",
        #                             "help": "Event-aligned average sub-population activity "
        #                             "line plot",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "event_aligned_subpopulation_activity_comparison"
        #                             ".preview.svg",
        #                             "file_format": "svg",
        #                         },
        #                         {
        #                             "name": "Fraction of neurons in each modulation group",
        #                             "help": "Fraction of neurons in each modulation group",
        #                             "file_path": "/tmp/tmp_combine_compare_peri_event_data/"
        #                             "modulated_fractions_comparison.preview.svg",
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
        #         expected_output_manifest=expected_output_manifest,
        #         tool_metadata=self.tool_metadata[self.tool_key],
        #         actual_manifest_data_filepath=self.output_manifest_file,
        #     )
        # )

    def test_combine_peri_event_group_names_with_spaces(self):
        # ensure output files do not contain spaces introduced by user-specified group names
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

        for group1_name, group2_name in [
            ["group1", "group2"],
            ["group 1", "group2"],
            ["group1", "group 2"],
            ["Group 1", "group 2"],
        ]:
            # combine data
            combine_compare_peri_event_data(
                group1_traces_files=group1_traces_files,
                group1_stats_files=group1_stats_files,
                group1_name=group1_name,
                group2_traces_files=group2_traces_files,
                group2_stats_files=group2_stats_files,
                group2_name=group2_name,
                comparison_type="two_tailed",
                data_pairing="paired",
                significance_threshold=None,
                average_method="neurons",
                tolerance=0.01,
                output_dir=self.output_dir,
            )

            # validate existence of expected output files
            group1_name = group1_name.replace(" ", "_")
            group2_name = group2_name.replace(" ", "_")
            actual_files = os.listdir(self.output_dir)
            expected_files = [
                # output files expected by the system
                # "output_manifest.json",
                "output_metadata.json",
                # output files from combining data in group 1
                f"event_aligned_activity_{group1_name}.csv",
                f"event_aligned_activity_by_modulation_{group1_name}.preview.svg",
                f"fraction_of_modulated_neurons_{group1_name}.preview.svg",
                f"event_aligned_statistics_{group1_name}.csv",
                f"event_aligned_population_activity_{group1_name}.preview.svg",
                f"event_aligned_single_cell_activity_heatmap_{group1_name}.preview.svg",
                # output files from combining data in group 2
                f"event_aligned_activity_{group2_name}.csv",
                f"event_aligned_activity_by_modulation_{group2_name}.preview.svg",
                f"fraction_of_modulated_neurons_{group2_name}.preview.svg",
                f"event_aligned_statistics_{group2_name}.csv",
                f"event_aligned_population_activity_{group2_name}.preview.svg",
                f"event_aligned_single_cell_activity_heatmap_{group2_name}.preview.svg",
                # comparison output files
                "comparison_data.csv",
                "event_aligned_population_activity_comparison.preview.svg",
                "event_aligned_subpopulation_activity_comparison.preview.svg",
                "modulated_fractions_comparison.preview.svg",
            ]
            self.assertEqual(sorted(actual_files), sorted(expected_files))

            # remove output files ahead of next iteration
            [
                os.remove(os.path.join(self.output_dir, f))
                for f in expected_files
            ]

    def test_combine_single_group_invalid_average_method(self):
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
        self.assertRaisesRegex(
            IdeasError,
            "An invalid average method was supplied. Must be 'neurons' or 'recordings'.",
            combine_compare_peri_event_data,
            group1_traces_files=group1_traces_files,
            group1_stats_files=group1_stats_files,
            group1_name="Group 1",
            group2_traces_files=[],
            group2_stats_files=[],
            group2_name="",
            comparison_type=None,
            data_pairing=None,
            significance_threshold=None,
            average_method="mixed",
            tolerance=0.01,
            output_dir=self.output_dir,
        )

    def test_combine_single_group_average_data_by_recordings(self):
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
        combine_compare_peri_event_data(
            group1_traces_files=group1_traces_files,
            group1_stats_files=group1_stats_files,
            group1_name=group_name,
            group2_traces_files=[],
            group2_stats_files=[],
            group2_name="",
            comparison_type=None,
            data_pairing=None,
            significance_threshold=None,
            average_method="recordings",
            tolerance=0.01,
            output_dir=self.output_dir,
        )

        # validate existence of expected output files
        actual_files = os.listdir(self.output_dir)
        expected_files = [
            # "output_manifest.json",
            "output_metadata.json",
            f"event_aligned_activity_{group_name}.csv",
            f"event_aligned_activity_by_modulation_{group_name}.preview.svg",
            f"fraction_of_modulated_neurons_{group_name}.preview.svg",
            f"event_aligned_statistics_{group_name}.csv",
            f"event_aligned_population_activity_{group_name}.preview.svg",
            f"event_aligned_single_cell_activity_heatmap_{group_name}.preview.svg",
        ]
        self.assertEqual(sorted(actual_files), sorted(expected_files))

        # validate event-aligned traces
        df_traces = pd.read_csv(
            os.path.join(
                self.output_dir, f"event_aligned_activity_{group_name}.csv"
            )
        )
        self.assertEqual(df_traces.shape, (119, 249))

        # validate event-aligned statistics
        df_stats = pd.read_csv(
            os.path.join(
                self.output_dir, f"event_aligned_statistics_{group_name}.csv"
            )
        )
        self.assertEqual(df_stats.shape, (240, 8))

    def test_combine_two_groups_average_data_by_recordings(self):
        # define input files
        group1_name = "group1"
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

        group2_name = "group2"
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
        combine_compare_peri_event_data(
            group1_traces_files=group1_traces_files,
            group1_stats_files=group1_stats_files,
            group1_name=group1_name,
            group2_traces_files=group2_traces_files,
            group2_stats_files=group2_stats_files,
            group2_name=group2_name,
            comparison_type="two_tailed",
            data_pairing="paired",
            significance_threshold=None,
            average_method="recordings",
            tolerance=0.01,
            output_dir=self.output_dir,
        )

        # validate existence of expected output files
        actual_files = os.listdir(self.output_dir)
        expected_files = [
            # output files expected by the system
            # "output_manifest.json",
            "output_metadata.json",
            # output files from combining data in group 1
            f"event_aligned_activity_{group1_name}.csv",
            f"event_aligned_activity_by_modulation_{group1_name}.preview.svg",
            f"fraction_of_modulated_neurons_{group1_name}.preview.svg",
            f"event_aligned_statistics_{group1_name}.csv",
            f"event_aligned_population_activity_{group1_name}.preview.svg",
            f"event_aligned_single_cell_activity_heatmap_{group1_name}.preview.svg",
            # output files from combining data in group 2
            f"event_aligned_activity_{group2_name}.csv",
            f"event_aligned_activity_by_modulation_{group2_name}.preview.svg",
            f"fraction_of_modulated_neurons_{group2_name}.preview.svg",
            f"event_aligned_statistics_{group2_name}.csv",
            f"event_aligned_population_activity_{group2_name}.preview.svg",
            f"event_aligned_single_cell_activity_heatmap_{group2_name}.preview.svg",
            # comparison output files
            "comparison_data.csv",
            "event_aligned_population_activity_comparison.preview.svg",
            "event_aligned_subpopulation_activity_comparison.preview.svg",
            "modulated_fractions_comparison.preview.svg",
        ]
        self.assertEqual(sorted(actual_files), sorted(expected_files))

        # validate event-aligned traces for GROUP 1
        df_traces_group1 = pd.read_csv(
            os.path.join(
                self.output_dir, f"event_aligned_activity_{group1_name}.csv"
            )
        )
        self.assertEqual(df_traces_group1.shape, (119, 249))

        # validate event-aligned statistics for GROUP 1
        df_stats_group1 = pd.read_csv(
            os.path.join(
                self.output_dir, f"event_aligned_statistics_{group1_name}.csv"
            )
        )
        self.assertEqual(df_stats_group1.shape, (240, 8))

        # validate event-aligned traces for GROUP 2
        df_traces_group2 = pd.read_csv(
            os.path.join(
                self.output_dir, f"event_aligned_activity_{group2_name}.csv"
            )
        )
        self.assertEqual(df_traces_group2.shape, (119, 249))

        # validate event-aligned statistics for GROUP 2
        df_stats_group2 = pd.read_csv(
            os.path.join(
                self.output_dir, f"event_aligned_statistics_{group2_name}.csv"
            )
        )
        self.assertEqual(df_stats_group2.shape, (240, 8))

    def test_combine_and_compare_invalid_group_names(self):
        # define input files
        group1_name = "g r !@#$$%^ 12"
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

        group2_name = "gr &^%$@* 67"
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
        combine_compare_peri_event_data(
            group1_traces_files=group1_traces_files,
            group1_stats_files=group1_stats_files,
            group1_name=group1_name,
            group2_traces_files=group2_traces_files,
            group2_stats_files=group2_stats_files,
            group2_name=group2_name,
            comparison_type="two_tailed",
            data_pairing="paired",
            significance_threshold=None,
            average_method="neurons",
            tolerance=0.01,
            output_dir=self.output_dir,
        )

        # validate existence of expected output files
        actual_files = os.listdir(self.output_dir)
        exp_group1_name = group1_name.replace("$", "�").replace(" ", "_")
        exp_group2_name = group2_name.replace("$", "�").replace(" ", "_")
        expected_files = [
            # output files expected by the system
            # "output_manifest.json",
            "output_metadata.json",
            # output files from combining data in group 1
            f"event_aligned_activity_{exp_group1_name}.csv",
            f"event_aligned_activity_by_modulation_{exp_group1_name}.preview.svg",
            f"fraction_of_modulated_neurons_{exp_group1_name}.preview.svg",
            f"event_aligned_statistics_{exp_group1_name}.csv",
            f"event_aligned_population_activity_{exp_group1_name}.preview.svg",
            f"event_aligned_single_cell_activity_heatmap_{exp_group1_name}.preview.svg",
            # output files from combining data in group 2
            f"event_aligned_activity_{exp_group2_name}.csv",
            f"event_aligned_activity_by_modulation_{exp_group2_name}.preview.svg",
            f"fraction_of_modulated_neurons_{exp_group2_name}.preview.svg",
            f"event_aligned_statistics_{exp_group2_name}.csv",
            f"event_aligned_population_activity_{exp_group2_name}.preview.svg",
            f"event_aligned_single_cell_activity_heatmap_{exp_group2_name}.preview.svg",
            # comparison output files
            "comparison_data.csv",
            "event_aligned_population_activity_comparison.preview.svg",
            "event_aligned_subpopulation_activity_comparison.preview.svg",
            "modulated_fractions_comparison.preview.svg",
        ]
        self.assertEqual(sorted(actual_files), sorted(expected_files))

    def test_combine_and_compare_peri_event_data_invalid_population_activity_plot_limits(
        self,
    ):
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

        for population_activity_plot_limits in [
            "",
            " ",
            "hello",
            "-1,xxx",
            "xxx,1.0",
            "x,y",
        ]:
            # execute workflow and expect a IdeasError error to be raised
            self.assertRaisesRegex(
                IdeasError,
                "Visualization parameter 'Population activity \(y-axis range\)' must be specified "
                "as 'auto' or 'min,max' \(e.g. -1,1\) where the minimum and "
                "maximum are not equal.",
                combine_compare_peri_event_data,
                group1_traces_files=group1_traces_files,
                group1_stats_files=group1_stats_files,
                group1_name="Group 1",
                group2_traces_files=group2_traces_files,
                group2_stats_files=group2_stats_files,
                group2_name="Group 2",
                comparison_type="two_tailed",
                data_pairing="paired",
                significance_threshold=0.05,
                average_method="neurons",
                tolerance=0.01,
                population_activity_plot_limits=population_activity_plot_limits,
            )

    def test_combine_and_compare_peri_event_data_invalid_activity_heatmap_color_limits(
        self,
    ):
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

        for activity_heatmap_color_limits in [
            "",
            " ",
            "hello",
            "-1,xxx",
            "xxx,1.0",
            "x,y",
        ]:
            # execute workflow and expect a IdeasError error to be raised
            self.assertRaisesRegex(
                IdeasError,
                "Visualization parameter 'Activity heatmap \(colormap range\)' must be specified "
                "as 'auto' or 'min,max' \(e.g. -1,1\) where the minimum and "
                "maximum are not equal.",
                combine_compare_peri_event_data,
                group1_traces_files=group1_traces_files,
                group1_stats_files=group1_stats_files,
                group1_name="Group 1",
                group2_traces_files=group2_traces_files,
                group2_stats_files=group2_stats_files,
                group2_name="Group 2",
                comparison_type="one_tailed_less",
                data_pairing="unpaired",
                significance_threshold=0.1,
                average_method="neurons",
                tolerance=0.02,
                activity_heatmap_color_limits=activity_heatmap_color_limits,
            )

    def test_combine_and_compare_peri_event_data_invalid_activity_by_modulation_plot_limits(
        self,
    ):
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

        for activity_by_modulation_plot_limits in [
            "",
            " ",
            "hello",
            "-1,xxx",
            "xxx,1.0",
            "x,y",
        ]:
            # execute workflow and expect a IdeasError error to be raised
            self.assertRaisesRegex(
                IdeasError,
                "Visualization parameter 'Activity by modulation group \(y-axis range\)' must be "
                "specified as 'auto' or 'min,max' \(e.g. -1,1\) where the "
                "minimum and maximum are not equal.",
                combine_compare_peri_event_data,
                group1_traces_files=group1_traces_files,
                group1_stats_files=group1_stats_files,
                group1_name="Group 1",
                group2_traces_files=group2_traces_files,
                group2_stats_files=group2_stats_files,
                group2_name="Group 2",
                comparison_type="one_tailed_less",
                data_pairing="unpaired",
                significance_threshold=0.1,
                average_method="neurons",
                tolerance=0.02,
                activity_by_modulation_plot_limits=activity_by_modulation_plot_limits,
            )

    def test_combine_single_group_with_plot_limits(self):
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
        combine_compare_peri_event_data(
            group1_traces_files=group1_traces_files,
            group1_stats_files=group1_stats_files,
            group1_name=group_name,
            group2_traces_files=[],
            group2_stats_files=[],
            group2_name="",
            comparison_type=None,
            data_pairing=None,
            significance_threshold=None,
            average_method="neurons",
            tolerance=0.05,
            output_dir=self.output_dir,
            population_activity_plot_limits="-0.15,0.23",
            activity_heatmap_color_limits="-0.13,0.45",
            activity_by_modulation_plot_limits="0,0.16",
        )

        # validate existence of expected output files
        actual_files = os.listdir(self.output_dir)
        expected_files = [
            # "output_manifest.json",
            "output_metadata.json",
            f"event_aligned_activity_{group_name}.csv",
            f"event_aligned_activity_by_modulation_{group_name}.preview.svg",
            f"fraction_of_modulated_neurons_{group_name}.preview.svg",
            f"event_aligned_statistics_{group_name}.csv",
            f"event_aligned_population_activity_{group_name}.preview.svg",
            f"event_aligned_single_cell_activity_heatmap_{group_name}.preview.svg",
        ]
        self.assertEqual(sorted(actual_files), sorted(expected_files))

    def test_combine_compare_two_groups_with_plot_limits(self):
        # define input files
        group1_name = "group1"
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

        group2_name = "group2"
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
        combine_compare_peri_event_data(
            group1_traces_files=group1_traces_files,
            group1_stats_files=group1_stats_files,
            group1_name=group1_name,
            group2_traces_files=group2_traces_files,
            group2_stats_files=group2_stats_files,
            group2_name=group2_name,
            comparison_type="one_tailed_greater",
            data_pairing="unpaired",
            significance_threshold=0.07,
            average_method="recordings",
            tolerance=0.07,
            output_dir=self.output_dir,
            population_activity_plot_limits="-0.15,0",
            activity_heatmap_color_limits="0,0.56",
            activity_by_modulation_plot_limits="-0.9,2.3",
        )

        # validate existence of expected output files
        actual_files = os.listdir(self.output_dir)
        expected_files = [
            # output files expected by the system
            # "output_manifest.json",
            "output_metadata.json",
            # output files from combining data in group 1
            f"event_aligned_activity_{group1_name}.csv",
            f"event_aligned_activity_by_modulation_{group1_name}.preview.svg",
            f"fraction_of_modulated_neurons_{group1_name}.preview.svg",
            f"event_aligned_statistics_{group1_name}.csv",
            f"event_aligned_population_activity_{group1_name}.preview.svg",
            f"event_aligned_single_cell_activity_heatmap_{group1_name}.preview.svg",
            # output files from combining data in group 2
            f"event_aligned_activity_{group2_name}.csv",
            f"event_aligned_activity_by_modulation_{group2_name}.preview.svg",
            f"fraction_of_modulated_neurons_{group2_name}.preview.svg",
            f"event_aligned_statistics_{group2_name}.csv",
            f"event_aligned_population_activity_{group2_name}.preview.svg",
            f"event_aligned_single_cell_activity_heatmap_{group2_name}.preview.svg",
            # comparison output files
            "comparison_data.csv",
            "event_aligned_population_activity_comparison.preview.svg",
            "event_aligned_subpopulation_activity_comparison.preview.svg",
            "modulated_fractions_comparison.preview.svg",
        ]
        self.assertEqual(sorted(actual_files), sorted(expected_files))

    def test_combine_single_group_and_set_second_group_name_to_none(self):
        # define input files
        group1_name = "D2 -Tone5"
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
        combine_compare_peri_event_data(
            group1_traces_files=group1_traces_files,
            group1_stats_files=group1_stats_files,
            group1_name=group1_name,
            group2_traces_files=None,
            group2_stats_files=None,
            group2_name=None,
            average_method="neurons",
            tolerance=0.05,
            comparison_type=None,
            data_pairing=None,
            significance_threshold=None,
            output_dir=self.output_dir,
            population_activity_plot_limits="auto",
            activity_heatmap_color_limits="auto",
            activity_by_modulation_plot_limits="auto",
        )

        # validate existence of expected output files
        group1_name = group1_name.replace(" ", "_")
        actual_files = os.listdir(self.output_dir)
        expected_files = [
            # output files expected by the system
            # "output_manifest.json",
            "output_metadata.json",
            # output files from combining data in group 1
            f"event_aligned_activity_{group1_name}.csv",
            f"event_aligned_activity_by_modulation_{group1_name}.preview.svg",
            f"fraction_of_modulated_neurons_{group1_name}.preview.svg",
            f"event_aligned_statistics_{group1_name}.csv",
            f"event_aligned_population_activity_{group1_name}.preview.svg",
            f"event_aligned_single_cell_activity_heatmap_{group1_name}.preview.svg",
        ]
        self.assertEqual(sorted(actual_files), sorted(expected_files))
