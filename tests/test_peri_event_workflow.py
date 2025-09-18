import json
import os
import shutil
import h5py
import unittest
import numpy as np
import pandas as pd
from analysis.peri_event_workflow import (
    peri_event_single_cell_analysis,
    run_peri_event_workflow,
    nanmean_iterative,
)
from ideas.exceptions import IdeasError
from utils.testing_utils import (
    compare_float_dataframes,
    # validate_json_schema,
    # validate_output_manifest,
)


class TestPeriEventWorkflow(unittest.TestCase):
    """Tests for the peri-event analysis workflow."""

    # tool info
    tool_key = "peri_event_workflow"

    # define directories
    temporary_dir = "/tmp"
    input_dir = "data/peri_event_workflow"
    output_dir = os.path.join(temporary_dir, "tmp_peri_event_workflow_outputs")

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

    # VALID CASES
    def test_peri_event_workflow_single_cell_set_single_event_type(self):
        # define input parameters
        input_parameters = {
            "event_types": ["event_type_1"],
            "visual_window": {"pre": -2, "post": 2},
            "statistical_window": {"pre": [-1, 0], "post": [0, 1]},
            "num_shuffles": 50,
            "significance_threshold": 0.05,
            "seed": 0,
        }

        # execute workflow
        run_peri_event_workflow(
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
            output_dir=self.output_dir,
        )

        # retrieve event type and define event-type-specific output dir
        self.assertTrue(len(input_parameters["event_types"]) == 1)
        event_type = input_parameters["event_types"][0]
        output_dir = os.path.join(self.output_dir, "event_type_" + event_type)

        # ensure expected CSV files exist
        traces_csv_filename = os.path.join(
            output_dir, "event_aligned_activity.TRACES.csv"
        )
        stats_csv_filename = os.path.join(
            output_dir, "event_aligned_activity.STATISTICS.csv"
        )
        for f in [traces_csv_filename, stats_csv_filename]:
            self.assertTrue(os.path.exists(f))

        # verify event-aligned TRACES data
        actual_traces_df = pd.read_csv(traces_csv_filename)
        self.assertTrue(
            len(actual_traces_df) == 41 and len(actual_traces_df.columns) == 52
        )

        expected_traces_df = pd.read_csv(
            os.path.join(self.input_dir, "expected_event_aligned_traces.csv")
        )
        compare_float_dataframes(actual_traces_df, expected_traces_df)

        # verify STATISTICS data
        actual_stats_df = pd.read_csv(stats_csv_filename)
        self.assertTrue(
            len(actual_stats_df) == 24 and len(actual_stats_df.columns) == 7
        )

        expected_stats_df = pd.read_csv(
            os.path.join(
                self.input_dir, "expected_event_aligned_statistics.csv"
            )
        )
        compare_float_dataframes(actual_stats_df, expected_stats_df)

        # define basename for output files
        population_activity_basename = os.path.join(
            output_dir, "event_aligned_population_activity"
        )
        modulation_basename = os.path.join(
            output_dir, "event_aligned_activity_by_modulation"
        )
        heatmap_basename = os.path.join(
            output_dir, "event_aligned_single_cell_activity_heatmap"
        )
        cell_map_basename = os.path.join(output_dir, "cell_map")

        # ensure PREVIEW files exist
        population_activity_preview_filename = (
            population_activity_basename + ".preview.svg"
        )
        modulation_preview_filename = modulation_basename + ".preview.svg"
        heatmap_preview_filename = heatmap_basename + ".preview.svg"
        cell_map_preview_filename = cell_map_basename + ".preview.svg"
        for f in [
            population_activity_preview_filename,
            modulation_preview_filename,
            heatmap_preview_filename,
            cell_map_preview_filename,
        ]:
            self.assertTrue(os.path.exists(f))

        # define expected output manifest
        expected_output_manifest = {
            "schema_version": "2.0.0",
            "groups": [
                {
                    "group_key": "peri_event_analysis_output",
                    "group_type": "tool_output",
                    "group_id": "21ce294e-e7bc-4939-b234-4ad311ed2ac3",
                    "series": [],
                    "files": [
                        {
                            "file_key": "input_cellset_files",
                            "file_name": "cellset_series_part1-PCA-ICA.isxd",
                            "file_id": "8da1b69a-c537-4bfc-af3d-536740d93491",
                            "file_path": "/ideas/toolbox/tests/data/"
                            "peri_event_workflow/"
                            "cellset_series_part1-PCA-ICA.isxd",
                            "file_type": "cell_set",
                            "file_format": "isxd",
                            "file_structure": "binary",
                            "file_category": "source",
                        },
                        {
                            "file_key": "input_events_h5_file",
                            "file_name": "events.h5",
                            "file_id": "afdf1760-e0ae-46b3-aad2-30bf295b4ff5",
                            "file_path": "/ideas/toolbox/tests/data/"
                            "peri_event_workflow/events.h5",
                            "file_type": "timestamp_events",
                            "file_format": "h5",
                            "file_structure": "sparse_time_series",
                            "file_category": "source",
                        },
                        {
                            "file_key": "event_aligned_activity_traces_event"
                            "_aligned_activity_traces_csv_file",
                            "file_name": "event_aligned_activity.TRACES.csv",
                            "file_id": "167f76fb-3bd0-4745-8681-4daff0f6ef76",
                            "file_path": "/tmp/tmp_peri_event_workflow_outputs/"
                            "event_type_event_type_1/event_aligned_activity"
                            ".TRACES.csv",
                            "file_type": "event_aligned_neural_data",
                            "file_format": "csv",
                            "file_structure": "time_series",
                            "file_category": "result",
                            "parent_ids": [
                                "8da1b69a-c537-4bfc-af3d-536740d93491"
                            ],
                            "preview": [
                                {
                                    "name": "Event-aligned population activity figure",
                                    "help": "Event-aligned average population activity line plot",
                                    "file_path": "/tmp/tmp_peri_event_workflow_outputs/"
                                    "event_type_event_type_1/event_aligned_"
                                    "population_activity.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned single-cell activity figure",
                                    "help": "Event-aligned single-cell activity heatmap",
                                    "file_path": "/tmp/tmp_peri_event_workflow_outputs/"
                                    "event_type_event_type_1/event_aligned_"
                                    "single_cell_activity_heatmap.preview.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                        {
                            "file_key": "event_aligned_statistics_event_"
                            "aligned_activity_statistics_csv_file",
                            "file_name": "event_aligned_activity.STATISTICS.csv",
                            "file_id": "971617a9-01c4-4d9b-8224-2c22199f30b2",
                            "file_path": "/tmp/tmp_peri_event_workflow_outputs/"
                            "event_type_event_type_1/event_aligned"
                            "_activity.STATISTICS.csv",
                            "file_type": "statistics",
                            "file_format": "csv",
                            "file_structure": "table",
                            "file_category": "result",
                            "parent_ids": [
                                "8da1b69a-c537-4bfc-af3d-536740d93491",
                                "167f76fb-3bd0-4745-8681-4daff0f6ef76",
                            ],
                            "preview": [
                                {
                                    "name": "Event-aligned sub-population activity figure",
                                    "help": "Event-aligned average sub-population activity"
                                    " line plot (up-, down-, and non-modulated neurons)",
                                    "file_path": "/tmp/tmp_peri_event_workflow_outputs/"
                                    "event_type_event_type_1/event_aligned_"
                                    "activity_by_modulation.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Spatial organization of modulation",
                                    "help": "Cell map visualizing spatial organization of "
                                    "modulation",
                                    "file_path": "/tmp/tmp_peri_event_workflow_outputs/"
                                    "event_type_event_type_1/cell_map.preview.svg",
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
        #         expected_output_manifest=expected_output_manifest,
        #         tool_metadata=self.tool_metadata[self.tool_key],
        #         actual_manifest_data_filepath=self.output_manifest_file,
        #     )
        # )

        # validate output metadata
        # self.assertTrue(os.path.exists(self.output_metadata_file))
        # validate_json_schema(
        #     self.output_metadata_file, self.output_metadata_json_schema
        # )

        # with open(self.output_metadata_file) as f:
        #     actual_metadata = json.load(f)
        #     group_metadata = actual_metadata["group_metadata"]
        #     series_metadata = actual_metadata["series_metadata"]
        #     file_metadata = actual_metadata["file_metadata"]

        # # validate group metadata
        # self.assertEqual(len(group_metadata), 1)
        # self.assertEqual(
        #     group_metadata[0]["add"]["ideas"]["dataset"]["group_type"],
        #     "tool_output",
        # )

        # # validate series metadata
        # self.assertEqual(series_metadata, [])

        # # validate file metadata
        # self.assertEqual(len(file_metadata), 2)
        # for f in file_metadata:
        #     self.assertEqual(list(f["add"].keys()), ["ideas"])
        #     self.assertEqual(
        #         f["add"]["ideas"]["timingInfo"]["sampling_rate"], 10
        #     )

    def test_peri_event_single_cell_analysis_no_modulated_cells(self):
        # tests the case where there are zero modulated cells

        # parameters
        visual_window = (40, 40)
        statistical_window_indices = {
            "pre": np.arange(20, 41),
            "post": np.arange(40, 61),
        }
        x_values = np.linspace(-2, 2, 81)
        x_limits = (-2, 2)
        event_type = "event_type_1"
        significance_threshold = 0.05

        input_traces_parquet_file = os.path.join(
            self.input_dir, "traces.parquet"
        )

        df = pd.read_parquet(input_traces_parquet_file)

        # fake events
        event_indices = list(
            np.round(np.linspace(1000, len(df) - 1000, 100)).astype(int)
        )

        # fake shuffles
        event_indices_shuffles = [list(event_indices) for _ in range(10)]

        # footprints
        input_footprints_h5_file = os.path.join(
            self.input_dir, "footprints.h5"
        )

        f = h5py.File(input_footprints_h5_file, "r")
        footprints = f["footprints"]
        footprints = footprints[()]

        peri_event_single_cell_analysis(
            df.drop(columns=["Time"]) * 0,  # no modulated cells here!
            footprints,
            event_indices,
            event_indices_shuffles,
            visual_window,
            statistical_window_indices,
            significance_threshold,
            x_values,
            x_limits,
            event_type,
            self.output_dir,
        )

    def test_peri_event_workflow_cell_set_series(self):
        # subset of BrainMinds data (2019-11-20, first 3 cell sets)
        input_cellset_isxd_files = [
            os.path.join(self.input_dir, "cellset_series_part1-PCA-ICA.isxd"),
            os.path.join(self.input_dir, "cellset_series_part2-PCA-ICA.isxd"),
        ]

        input_parameters = {
            "event_types": ["event_type_1"],
            "visual_window": {"pre": -2, "post": 2},
            "statistical_window": {"pre": [-1, 0], "post": [0, 1]},
            "num_shuffles": 100,
            "significance_threshold": 0.05,
            "seed": 0,
        }

        # execute workflow
        run_peri_event_workflow(
            input_cellset_files=input_cellset_isxd_files,
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
            output_dir=self.output_dir,
        )

        # retrieve event type and define event-type-specific output dir
        self.assertTrue(len(input_parameters["event_types"]) == 1)
        event_type = input_parameters["event_types"][0]
        output_dir = os.path.join(self.output_dir, "event_type_" + event_type)

        # ensure expected CSV files exist
        traces_csv_filename = os.path.join(
            output_dir, "event_aligned_activity.TRACES.csv"
        )
        stats_csv_filename = os.path.join(
            output_dir, "event_aligned_activity.STATISTICS.csv"
        )
        for f in [traces_csv_filename, stats_csv_filename]:
            self.assertTrue(os.path.exists(f))

        # verify event-aligned TRACES data
        actual_traces_df = pd.read_csv(traces_csv_filename)
        # rows: number of timepoints per window
        # columns: number of traces (2 per cell [mean, sem],
        # 2 per group [population, up, down, non], 1 for Time)
        self.assertTrue(
            len(actual_traces_df) == 41 and len(actual_traces_df.columns) == 52
        )

        expected_traces_df = pd.read_csv(
            os.path.join(
                self.input_dir,
                "expected_series_event_aligned_activity.TRACES.csv",
            )
        )
        compare_float_dataframes(actual_traces_df, expected_traces_df)

        # verify STATISTICS table
        actual_stats_df = pd.read_csv(stats_csv_filename)
        # rows: 1 per cell and 1 per group (population, up, down, non)
        # columns: 1 per statistics
        self.assertTrue(
            len(actual_stats_df) == 24 and len(actual_stats_df.columns) == 7
        )

        expected_stats_df = pd.read_csv(
            os.path.join(
                self.input_dir,
                "expected_series_event_aligned_activity.STATISTICS.csv",
            )
        )
        compare_float_dataframes(actual_stats_df, expected_stats_df)

        # define basename for output files
        population_activity_basename = os.path.join(
            output_dir, "event_aligned_population_activity"
        )
        modulation_basename = os.path.join(
            output_dir, "event_aligned_activity_by_modulation"
        )
        heatmap_basename = os.path.join(
            output_dir, "event_aligned_single_cell_activity_heatmap"
        )
        cell_map_basename = os.path.join(output_dir, "cell_map")

        # ensure PREVIEW files exist
        population_activity_preview_filename = (
            population_activity_basename + ".preview.svg"
        )
        modulation_preview_filename = modulation_basename + ".preview.svg"
        heatmap_preview_filename = heatmap_basename + ".preview.svg"
        cell_map_preview_filename = cell_map_basename + ".preview.svg"
        for f in [
            population_activity_preview_filename,
            modulation_preview_filename,
            heatmap_preview_filename,
            cell_map_preview_filename,
        ]:
            self.assertTrue(os.path.exists(f))

        # define expected output manifest
        expected_output_manifest = {
            "schema_version": "2.0.0",
            "groups": [
                {
                    "group_key": "peri_event_analysis_output",
                    "group_type": "tool_output",
                    "group_id": "9cf442c9-210c-4e23-8e71-052d6e7d30bb",
                    "series": [],
                    "files": [
                        {
                            "file_key": "input_cellset_files",
                            "file_name": "cellset_series_part1-PCA-ICA.isxd",
                            "file_id": "c3d70d60-961b-45ec-a3c1-2b88e9d5ca64",
                            "file_path": "/ideas/toolbox/tests/data/peri_event_workflow/"
                            "cellset_series_part1-PCA-ICA.isxd",
                            "file_type": "cell_set",
                            "file_format": "isxd",
                            "file_structure": "binary",
                            "file_category": "source",
                        },
                        {
                            "file_key": "input_cellset_files",
                            "file_name": "cellset_series_part2-PCA-ICA.isxd",
                            "file_id": "f3ed71fc-2a6c-4806-bd7b-bec35a29c606",
                            "file_path": "/ideas/toolbox/tests/data/peri_event_workflow"
                            "/cellset_series_part2-PCA-ICA.isxd",
                            "file_type": "cell_set",
                            "file_format": "isxd",
                            "file_structure": "binary",
                            "file_category": "source",
                        },
                        {
                            "file_key": "input_events_h5_file",
                            "file_name": "events.h5",
                            "file_id": "6ae088ef-0fd0-4140-9736-024c8d555044",
                            "file_path": "/ideas/toolbox/tests/data/"
                            "peri_event_workflow/events.h5",
                            "file_type": "timestamp_events",
                            "file_format": "h5",
                            "file_structure": "sparse_time_series",
                            "file_category": "source",
                        },
                        {
                            "file_key": "event_aligned_activity_traces_event"
                            "_aligned_activity_traces_csv_file",
                            "file_name": "event_aligned_activity.TRACES.csv",
                            "file_id": "4685184a-20cd-4c61-b071-b7fb7facf676",
                            "file_path": "/tmp/tmp_peri_event_workflow_outputs/"
                            "event_type_event_type_1/event_aligned"
                            "_activity.TRACES.csv",
                            "file_type": "event_aligned_neural_data",
                            "file_format": "csv",
                            "file_structure": "time_series",
                            "file_category": "result",
                            "parent_ids": [
                                "c3d70d60-961b-45ec-a3c1-2b88e9d5ca64",
                                "f3ed71fc-2a6c-4806-bd7b-bec35a29c606",
                            ],
                            "preview": [
                                {
                                    "name": "Event-aligned population activity "
                                    "figure",
                                    "help": "Event-aligned average population "
                                    "activity line plot",
                                    "file_path": "/tmp/tmp_peri_event_workflow_"
                                    "outputs/event_type_event_type_1/"
                                    "event_aligned_population_activity"
                                    ".preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned single-cell activity figure",
                                    "help": "Event-aligned single-cell activity heatmap",
                                    "file_path": "/tmp/tmp_peri_event_workflow_outputs/"
                                    "event_type_event_type_1/event_aligned"
                                    "_single_cell_activity_heatmap.preview"
                                    ".svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                        {
                            "file_key": "event_aligned_statistics_event_aligned"
                            "_activity_statistics_csv_file",
                            "file_name": "event_aligned_activity.STATISTICS.csv",
                            "file_id": "bddb084e-a638-4c64-aa4b-0c6a677ab30d",
                            "file_path": "/tmp/tmp_peri_event_workflow_outputs/"
                            "event_type_event_type_1/event_aligned"
                            "_activity.STATISTICS.csv",
                            "file_type": "statistics",
                            "file_format": "csv",
                            "file_structure": "table",
                            "file_category": "result",
                            "parent_ids": [
                                "c3d70d60-961b-45ec-a3c1-2b88e9d5ca64",
                                "4685184a-20cd-4c61-b071-b7fb7facf676",
                            ],
                            "preview": [
                                {
                                    "name": "Event-aligned sub-population "
                                    "activity figure",
                                    "help": "Event-aligned average sub-population "
                                    "activity line plot (up-, down-, and "
                                    "non-modulated neurons)",
                                    "file_path": "/tmp/tmp_peri_event_workflow_outputs"
                                    "/event_type_event_type_1/event_aligned"
                                    "_activity_by_modulation.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Spatial organization of modulation",
                                    "help": "Cell map visualizing spatial organization"
                                    " of modulation",
                                    "file_path": "/tmp/tmp_peri_event_workflow_outputs"
                                    "/event_type_event_type_1/cell_map"
                                    ".preview.svg",
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
        #         expected_output_manifest=expected_output_manifest,
        #         tool_metadata=self.tool_metadata[self.tool_key],
        #         actual_manifest_data_filepath=self.output_manifest_file,
        #     )
        # )

        # validate output metadata
        # self.assertTrue(os.path.exists(self.output_metadata_file))
        # validate_json_schema(
        #     self.output_metadata_file, self.output_metadata_json_schema
        # )

        # with open(self.output_metadata_file) as f:
        #     actual_metadata = json.load(f)
        #     group_metadata = actual_metadata["group_metadata"]
        #     series_metadata = actual_metadata["series_metadata"]
        #     file_metadata = actual_metadata["file_metadata"]

        # # validate group metadata
        # self.assertEqual(len(group_metadata), 1)
        # self.assertEqual(
        #     group_metadata[0]["add"]["ideas"]["dataset"]["group_type"],
        #     "tool_output",
        # )

        # # validate series metadata
        # self.assertEqual(series_metadata, [])

        # # validate file metadata
        # self.assertEqual(len(file_metadata), 2)
        # for f in file_metadata:
        #     self.assertEqual(list(f["add"].keys()), ["ideas"])
        #     self.assertEqual(
        #         f["add"]["ideas"]["timingInfo"]["sampling_rate"], 10
        #     )

    # ERROR CASES
    def test_peri_event_workflow_missing_cellset_file(self):
        input_cellset_isxd_files = ["/tmp/nfiuanuiadfnain/qodqm/ssndnd.isxd"]
        input_parameters = {
            "event_types": ["event_type_1"],
            "visual_window": {"pre": -2, "post": 2},
            "statistical_window": {"pre": [-1, 0], "post": [0, 1]},
            "num_shuffles": 50,
            "significance_threshold": 0.05,
            "seed": 0,
        }

        self.assertRaises(
            IdeasError,
            run_peri_event_workflow,
            input_cellset_files=input_cellset_isxd_files,
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
            output_dir=self.output_dir,
        )

    def test_peri_event_workflow_missing_events_file(self):
        # define input parameters
        input_events_h5_file = "/tmp/nfiuanuiadfnain/qodqm/ssndnd_events.h5"
        input_parameters = {
            "event_types": ["event_type_1"],
            "visual_window": {"pre": -2, "post": 2},
            "statistical_window": {"pre": [-1, 0], "post": [0, 1]},
            "num_shuffles": 50,
            "significance_threshold": 0.05,
            "seed": 0,
        }

        self.assertRaises(
            IdeasError,
            run_peri_event_workflow,
            input_cellset_files=self.input_cellset_isxd_files,
            input_events_h5_file=input_events_h5_file,
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
            output_dir=self.output_dir,
        )

    def test_peri_event_workflow_no_event_of_specified_type(self):
        # define input parameters
        input_parameters = {
            "event_types": ["fake_event_123"],
            "visual_window": {"pre": -2, "post": 2},
            "statistical_window": {"pre": [-1, 0], "post": [0, 1]},
            "num_shuffles": 50,
            "significance_threshold": 0.05,
            "seed": 0,
        }

        # execute workflow and expect a IdeasError error to be raised
        self.assertRaises(
            IdeasError,
            run_peri_event_workflow,
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
            output_dir=self.output_dir,
        )

    def test_peri_event_workflow_invalid_significance_threshold(self):
        # define input parameters
        input_parameters = {
            "event_types": ["event_type_1"],
            "visual_window": {"pre": -2, "post": 2},
            "statistical_window": {"pre": [-1, 0], "post": [0, 1]},
            "num_shuffles": 50,
            "seed": 0,
        }

        for significance_threshold in [-0.1, 1.2]:
            input_parameters["significance_threshold"] = significance_threshold

            # execute workflow and expect a IdeasError error to be raised
            self.assertRaises(
                IdeasError,
                run_peri_event_workflow,
                input_cellset_files=self.input_cellset_isxd_files,
                input_events_h5_file=self.input_events_h5_file,
                event_type=input_parameters["event_types"][0],
                visual_window_pre=input_parameters["visual_window"]["pre"],
                visual_window_post=input_parameters["visual_window"]["post"],
                statistical_window_pre_start=input_parameters[
                    "statistical_window"
                ]["pre"][0],
                statistical_window_pre_end=input_parameters[
                    "statistical_window"
                ]["pre"][1],
                statistical_window_post_start=input_parameters[
                    "statistical_window"
                ]["post"][0],
                statistical_window_post_end=input_parameters[
                    "statistical_window"
                ]["post"][1],
                num_shuffles=input_parameters["num_shuffles"],
                significance_threshold=input_parameters[
                    "significance_threshold"
                ],
                seed=input_parameters["seed"],
                output_dir=self.output_dir,
            )

    def test_peri_event_workflow_invalid_number_of_shuffles(self):
        # define input parameters
        input_parameters = {
            "event_types": ["event_type_1"],
            "visual_window": {"pre": -2, "post": 2},
            "statistical_window": {"pre": [-1, 0], "post": [0, 1]},
            "significance_threshold": 0.05,
            "seed": 0,
        }

        for num_shuffles in [-1, 0, 1.0]:
            input_parameters["num_shuffles"] = num_shuffles

            # execute workflow and expect a IdeasError error to be raised
            self.assertRaises(
                IdeasError,
                run_peri_event_workflow,
                input_cellset_files=self.input_cellset_isxd_files,
                input_events_h5_file=self.input_events_h5_file,
                event_type=input_parameters["event_types"][0],
                visual_window_pre=input_parameters["visual_window"]["pre"],
                visual_window_post=input_parameters["visual_window"]["post"],
                statistical_window_pre_start=input_parameters[
                    "statistical_window"
                ]["pre"][0],
                statistical_window_pre_end=input_parameters[
                    "statistical_window"
                ]["pre"][1],
                statistical_window_post_start=input_parameters[
                    "statistical_window"
                ]["post"][0],
                statistical_window_post_end=input_parameters[
                    "statistical_window"
                ]["post"][1],
                num_shuffles=input_parameters["num_shuffles"],
                significance_threshold=input_parameters[
                    "significance_threshold"
                ],
                seed=input_parameters["seed"],
                output_dir=self.output_dir,
            )

    def test_peri_event_workflow_visual_window_of_size_zero(self):
        # define input parameters
        input_parameters = {
            "event_types": ["event_type_1"],
            "visual_window": {"pre": 0, "post": 0},
            "statistical_window": {"pre": [-1, 0], "post": [0, 1]},
            "num_shuffles": 50,
            "significance_threshold": 0.05,
            "seed": 0,
        }

        # execute workflow and expect a IdeasError error to be raised
        self.assertRaises(
            IdeasError,
            run_peri_event_workflow,
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
            output_dir=self.output_dir,
        )

    def test_peri_event_workflow_invalid_visual_windows(self):
        # define input parameters
        input_parameters = {
            "event_types": ["event_type_1"],
            "statistical_window": {"pre": [-1, 0], "post": [0, 1]},
            "num_shuffles": 50,
            "significance_threshold": 0.05,
            "seed": 0,
        }

        for pre, post in [
            (1, 2),  # invalid pre
            (1, -1),  # invalid post
            (-1, -1),  # identical pre and post
        ]:
            input_parameters["visual_window"] = {"pre": pre, "post": post}

            # execute workflow and expect a IdeasError error to be raised
            self.assertRaises(
                IdeasError,
                run_peri_event_workflow,
                input_cellset_files=self.input_cellset_isxd_files,
                input_events_h5_file=self.input_events_h5_file,
                event_type=input_parameters["event_types"][0],
                visual_window_pre=input_parameters["visual_window"]["pre"],
                visual_window_post=input_parameters["visual_window"]["post"],
                statistical_window_pre_start=input_parameters[
                    "statistical_window"
                ]["pre"][0],
                statistical_window_pre_end=input_parameters[
                    "statistical_window"
                ]["pre"][1],
                statistical_window_post_start=input_parameters[
                    "statistical_window"
                ]["post"][0],
                statistical_window_post_end=input_parameters[
                    "statistical_window"
                ]["post"][1],
                num_shuffles=input_parameters["num_shuffles"],
                significance_threshold=input_parameters[
                    "significance_threshold"
                ],
                seed=input_parameters["seed"],
                output_dir=self.output_dir,
            )

    def test_peri_event_workflow_invalid_statistical_windows(self):
        # define input parameters
        input_parameters = {
            "event_types": ["event_type_1"],
            "visual_window": {"pre": -2, "post": 2},
            "num_shuffles": 50,
            "significance_threshold": 0.05,
            "seed": 0,
        }

        for pre, post in [
            ([0, -1], [0, 2]),  # invalid pre
            ([-1, 0], [2, 0]),  # invalid post
            ([-1, 0], [-1, 0]),  # identical pre and post windows
            ([0, 0], [0, 0]),  # empty statistical window
        ]:
            input_parameters["statistical_window"] = {"pre": pre, "post": post}

            # execute workflow and expect a IdeasError error to be raised
            self.assertRaises(
                IdeasError,
                run_peri_event_workflow,
                input_cellset_files=self.input_cellset_isxd_files,
                input_events_h5_file=self.input_events_h5_file,
                event_type=input_parameters["event_types"][0],
                visual_window_pre=input_parameters["visual_window"]["pre"],
                visual_window_post=input_parameters["visual_window"]["post"],
                statistical_window_pre_start=input_parameters[
                    "statistical_window"
                ]["pre"][0],
                statistical_window_pre_end=input_parameters[
                    "statistical_window"
                ]["pre"][1],
                statistical_window_post_start=input_parameters[
                    "statistical_window"
                ]["post"][0],
                statistical_window_post_end=input_parameters[
                    "statistical_window"
                ]["post"][1],
                num_shuffles=input_parameters["num_shuffles"],
                significance_threshold=input_parameters[
                    "significance_threshold"
                ],
                seed=input_parameters["seed"],
                output_dir=self.output_dir,
            )

    def test_peri_event_workflow_statistical_window_extends_beyond_visual_window(
        self,
    ):
        # define input parameters
        input_parameters = {
            "event_types": ["event_type_1"],
            "visual_window": {"pre": -2, "post": 2},
            "num_shuffles": 50,
            "significance_threshold": 0.05,
            "seed": 0,
        }

        for stats_pre, stats_post in [
            ([-4, 0], [0, 2]),  # statistical pre begins before visual pre
            ([-3, -1], [1, 2.5]),  # statistical post ends after visual post
            ([-4, 0], [1, 2.5]),  # both cases above combined
        ]:
            # update visual and statistical window parameters
            input_parameters["statistical_window"] = {"pre": -3, "post": 2}
            input_parameters["statistical_window"] = {
                "pre": stats_pre,
                "post": stats_post,
            }

            # execute workflow and expect a IdeasError error to be raised
            self.assertRaises(
                IdeasError,
                run_peri_event_workflow,
                input_cellset_files=self.input_cellset_isxd_files,
                input_events_h5_file=self.input_events_h5_file,
                event_type=input_parameters["event_types"][0],
                visual_window_pre=input_parameters["visual_window"]["pre"],
                visual_window_post=input_parameters["visual_window"]["post"],
                statistical_window_pre_start=input_parameters[
                    "statistical_window"
                ]["pre"][0],
                statistical_window_pre_end=input_parameters[
                    "statistical_window"
                ]["pre"][1],
                statistical_window_post_start=input_parameters[
                    "statistical_window"
                ]["post"][0],
                statistical_window_post_end=input_parameters[
                    "statistical_window"
                ]["post"][1],
                num_shuffles=input_parameters["num_shuffles"],
                significance_threshold=input_parameters[
                    "significance_threshold"
                ],
                seed=input_parameters["seed"],
                output_dir=self.output_dir,
            )

    def test_peri_event_workflow_invalid_seed(self):
        # define input parameters
        input_parameters = {
            "event_types": ["event_type_1"],
            "visual_window": {"pre": -2, "post": 2},
            "statistical_window": {"pre": [-1, 0], "post": [0, 1]},
            "num_shuffles": 50,
            "significance_threshold": 0.05,
        }

        for seed in [-1, 0.1, 2**32]:
            input_parameters["seed"] = seed

            # execute workflow and expect a IdeasError error to be raised
            self.assertRaises(
                IdeasError,
                run_peri_event_workflow,
                input_cellset_files=self.input_cellset_isxd_files,
                input_events_h5_file=self.input_events_h5_file,
                event_type=input_parameters["event_types"][0],
                visual_window_pre=input_parameters["visual_window"]["pre"],
                visual_window_post=input_parameters["visual_window"]["post"],
                statistical_window_pre_start=input_parameters[
                    "statistical_window"
                ]["pre"][0],
                statistical_window_pre_end=input_parameters[
                    "statistical_window"
                ]["pre"][1],
                statistical_window_post_start=input_parameters[
                    "statistical_window"
                ]["post"][0],
                statistical_window_post_end=input_parameters[
                    "statistical_window"
                ]["post"][1],
                num_shuffles=input_parameters["num_shuffles"],
                significance_threshold=input_parameters[
                    "significance_threshold"
                ],
                seed=input_parameters["seed"],
                output_dir=self.output_dir,
            )

    def test_peri_event_workflow_invalid_population_activity_plot_limits(self):
        input_parameters = {
            "event_types": ["event_type_1"],
            "visual_window": {"pre": -2, "post": 2},
            "statistical_window": {"pre": [-2, 0], "post": [0, 2]},
            "num_shuffles": 50,
            "significance_threshold": 0.05,
            "seed": 0,
        }

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
                run_peri_event_workflow,
                input_cellset_files=self.input_cellset_isxd_files,
                input_events_h5_file=self.input_events_h5_file,
                event_type=input_parameters["event_types"][0],
                visual_window_pre=input_parameters["visual_window"]["pre"],
                visual_window_post=input_parameters["visual_window"]["post"],
                statistical_window_pre_start=input_parameters[
                    "statistical_window"
                ]["pre"][0],
                statistical_window_pre_end=input_parameters[
                    "statistical_window"
                ]["pre"][1],
                statistical_window_post_start=input_parameters[
                    "statistical_window"
                ]["post"][0],
                statistical_window_post_end=input_parameters[
                    "statistical_window"
                ]["post"][1],
                num_shuffles=input_parameters["num_shuffles"],
                significance_threshold=input_parameters[
                    "significance_threshold"
                ],
                seed=input_parameters["seed"],
                population_activity_plot_limits=population_activity_plot_limits,
                output_dir=self.output_dir,
            )

    def test_peri_event_workflow_invalid_activity_heatmap_color_limits(self):
        input_parameters = {
            "event_types": ["event_type_1"],
            "visual_window": {"pre": -2, "post": 2},
            "statistical_window": {"pre": [-2, 0], "post": [0, 2]},
            "num_shuffles": 50,
            "significance_threshold": 0.05,
            "seed": 0,
        }

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
                run_peri_event_workflow,
                input_cellset_files=self.input_cellset_isxd_files,
                input_events_h5_file=self.input_events_h5_file,
                event_type=input_parameters["event_types"][0],
                visual_window_pre=input_parameters["visual_window"]["pre"],
                visual_window_post=input_parameters["visual_window"]["post"],
                statistical_window_pre_start=input_parameters[
                    "statistical_window"
                ]["pre"][0],
                statistical_window_pre_end=input_parameters[
                    "statistical_window"
                ]["pre"][1],
                statistical_window_post_start=input_parameters[
                    "statistical_window"
                ]["post"][0],
                statistical_window_post_end=input_parameters[
                    "statistical_window"
                ]["post"][1],
                num_shuffles=input_parameters["num_shuffles"],
                significance_threshold=input_parameters[
                    "significance_threshold"
                ],
                seed=input_parameters["seed"],
                activity_heatmap_color_limits=activity_heatmap_color_limits,
                output_dir=self.output_dir,
            )

    def test_peri_event_workflow_invalid_activity_by_modulation_plot_limits(
        self,
    ):
        input_parameters = {
            "event_types": ["event_type_1"],
            "visual_window": {"pre": -2, "post": 2},
            "statistical_window": {"pre": [-2, 0], "post": [0, 2]},
            "num_shuffles": 50,
            "significance_threshold": 0.05,
            "seed": 0,
        }

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
                run_peri_event_workflow,
                input_cellset_files=self.input_cellset_isxd_files,
                input_events_h5_file=self.input_events_h5_file,
                event_type=input_parameters["event_types"][0],
                visual_window_pre=input_parameters["visual_window"]["pre"],
                visual_window_post=input_parameters["visual_window"]["post"],
                statistical_window_pre_start=input_parameters[
                    "statistical_window"
                ]["pre"][0],
                statistical_window_pre_end=input_parameters[
                    "statistical_window"
                ]["pre"][1],
                statistical_window_post_start=input_parameters[
                    "statistical_window"
                ]["post"][0],
                statistical_window_post_end=input_parameters[
                    "statistical_window"
                ]["post"][1],
                num_shuffles=input_parameters["num_shuffles"],
                significance_threshold=input_parameters[
                    "significance_threshold"
                ],
                seed=input_parameters["seed"],
                activity_by_modulation_plot_limits=activity_by_modulation_plot_limits,
                output_dir=self.output_dir,
            )

    def test_peri_event_workflow_single_event(self):
        # The peri-event workflow is valid even when supplied a single event.
        # Here we validate that the workflow completes successfully and works
        # as expected with data containing a single event, in this case a
        # single shock given to an animal subject at a particular time.

        # define input parameters
        event_type = "Shock_1"
        input_parameters = {
            "event_types": [event_type],
            "visual_window": {"pre": -20, "post": 20},
            "statistical_window": {"pre": [-20, 0], "post": [0, 20]},
            "num_shuffles": 78,
            "significance_threshold": 0.05,
            "seed": 0,
        }

        input_cellset_isxd_files = [
            os.path.join(self.input_dir, "cellset_series_part1-PCA-ICA.isxd")
        ]
        input_events_h5_file = os.path.join(self.input_dir, "single_event.h5")

        # ensure input files contains a single Shock 1 event
        h5_events_file = h5py.File(input_events_h5_file, "r")
        self.assertEqual(len(h5_events_file["events"][event_type]["Time"]), 1)
        h5_events_file.close()

        # execute workflow
        run_peri_event_workflow(
            input_cellset_files=input_cellset_isxd_files,
            input_events_h5_file=input_events_h5_file,
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
            output_dir=self.output_dir,
        )

        # retrieve event type and define event-type-specific output dir
        self.assertTrue(len(input_parameters["event_types"]) == 1)
        event_type = input_parameters["event_types"][0]
        output_dir = os.path.join(self.output_dir, "event_type_" + event_type)

        # ensure expected CSV files exist
        traces_csv_filename = os.path.join(
            output_dir, "event_aligned_activity.TRACES.csv"
        )
        stats_csv_filename = os.path.join(
            output_dir, "event_aligned_activity.STATISTICS.csv"
        )
        for f in [traces_csv_filename, stats_csv_filename]:
            self.assertTrue(os.path.exists(f))

        # verify event-aligned TRACES data
        actual_traces_df = pd.read_csv(traces_csv_filename)
        self.assertTrue(
            len(actual_traces_df) == 401
            and len(actual_traces_df.columns) == 52
        )

        # verify STATISTICS data
        actual_stats_df = pd.read_csv(stats_csv_filename)
        self.assertTrue(
            len(actual_stats_df) == 24 and len(actual_stats_df.columns) == 7
        )

        # define basename for output files
        population_activity_basename = os.path.join(
            output_dir, "event_aligned_population_activity"
        )
        modulation_basename = os.path.join(
            output_dir, "event_aligned_activity_by_modulation"
        )
        heatmap_basename = os.path.join(
            output_dir, "event_aligned_single_cell_activity_heatmap"
        )
        cell_map_basename = os.path.join(output_dir, "cell_map")

        # ensure PREVIEW files exist
        population_activity_preview_filename = (
            population_activity_basename + ".preview.svg"
        )
        modulation_preview_filename = modulation_basename + ".preview.svg"
        heatmap_preview_filename = heatmap_basename + ".preview.svg"
        cell_map_preview_filename = cell_map_basename + ".preview.svg"
        for f in [
            population_activity_preview_filename,
            modulation_preview_filename,
            heatmap_preview_filename,
            cell_map_preview_filename,
        ]:
            self.assertTrue(os.path.exists(f))

        # define expected output manifest
        expected_output_manifest = {
            "schema_version": "2.0.0",
            "groups": [
                {
                    "group_key": "peri_event_analysis_output",
                    "group_type": "tool_output",
                    "group_id": "21ce294e-e7bc-4939-b234-4ad311ed2ac3",
                    "series": [],
                    "files": [
                        {
                            "file_key": "input_cellset_files",
                            "file_name": "cellset_series_part1-PCA-ICA.isxd",
                            "file_id": "8da1b69a-c537-4bfc-af3d-536740d93491",
                            "file_path": "/ideas/toolbox/tests/data/"
                            "peri_event_workflow/"
                            "cellset_series_part1-PCA-ICA.isxd",
                            "file_type": "cell_set",
                            "file_format": "isxd",
                            "file_structure": "binary",
                            "file_category": "source",
                        },
                        {
                            "file_key": "input_events_h5_file",
                            "file_name": "single_event.h5",
                            "file_id": "afdf1760-e0ae-46b3-aad2-30bf295b4ff5",
                            "file_path": "/ideas/toolbox/tests/data/"
                            "peri_event_workflow/single_event.h5",
                            "file_type": "timestamp_events",
                            "file_format": "h5",
                            "file_structure": "sparse_time_series",
                            "file_category": "source",
                        },
                        {
                            "file_key": "event_aligned_activity_traces_event"
                            "_aligned_activity_traces_csv_file",
                            "file_name": "event_aligned_activity.TRACES.csv",
                            "file_id": "167f76fb-3bd0-4745-8681-4daff0f6ef76",
                            "file_path": "/tmp/tmp_peri_event_workflow_outputs/"
                            "event_type_Shock_1/event_aligned_activity"
                            ".TRACES.csv",
                            "file_type": "event_aligned_neural_data",
                            "file_format": "csv",
                            "file_structure": "time_series",
                            "file_category": "result",
                            "parent_ids": [
                                "8da1b69a-c537-4bfc-af3d-536740d93491"
                            ],
                            "preview": [
                                {
                                    "name": "Event-aligned population activity figure",
                                    "help": "Event-aligned average population activity line plot",
                                    "file_path": "/tmp/tmp_peri_event_workflow_outputs/"
                                    "event_type_Shock_1/event_aligned_"
                                    "population_activity.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Event-aligned single-cell activity figure",
                                    "help": "Event-aligned single-cell activity heatmap",
                                    "file_path": "/tmp/tmp_peri_event_workflow_outputs/"
                                    "event_type_Shock_1/event_aligned_"
                                    "single_cell_activity_heatmap.preview.svg",
                                    "file_format": "svg",
                                },
                            ],
                        },
                        {
                            "file_key": "event_aligned_statistics_event_"
                            "aligned_activity_statistics_csv_file",
                            "file_name": "event_aligned_activity.STATISTICS.csv",
                            "file_id": "971617a9-01c4-4d9b-8224-2c22199f30b2",
                            "file_path": "/tmp/tmp_peri_event_workflow_outputs/"
                            "event_type_Shock_1/event_aligned"
                            "_activity.STATISTICS.csv",
                            "file_type": "statistics",
                            "file_format": "csv",
                            "file_structure": "table",
                            "file_category": "result",
                            "parent_ids": [
                                "8da1b69a-c537-4bfc-af3d-536740d93491",
                                "167f76fb-3bd0-4745-8681-4daff0f6ef76",
                            ],
                            "preview": [
                                {
                                    "name": "Event-aligned sub-population activity figure",
                                    "help": "Event-aligned average sub-population activity"
                                    " line plot (up-, down-, and non-modulated neurons)",
                                    "file_path": "/tmp/tmp_peri_event_workflow_outputs/"
                                    "event_type_Shock_1/event_aligned_"
                                    "activity_by_modulation.preview.svg",
                                    "file_format": "svg",
                                },
                                {
                                    "name": "Spatial organization of modulation",
                                    "help": "Cell map visualizing spatial organization of "
                                    "modulation",
                                    "file_path": "/tmp/tmp_peri_event_workflow_outputs/"
                                    "event_type_Shock_1/cell_map.preview.svg",
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
        #         expected_output_manifest=expected_output_manifest,
        #         tool_metadata=self.tool_metadata[self.tool_key],
        #         actual_manifest_data_filepath=self.output_manifest_file,
        #     )
        # )

        # validate output metadata
        # self.assertTrue(os.path.exists(self.output_metadata_file))
        # validate_json_schema(
        #     self.output_metadata_file, self.output_metadata_json_schema
        # )

        # with open(self.output_metadata_file) as f:
        #     actual_metadata = json.load(f)
        #     group_metadata = actual_metadata["group_metadata"]
        #     series_metadata = actual_metadata["series_metadata"]
        #     file_metadata = actual_metadata["file_metadata"]

        # # validate group metadata
        # self.assertEqual(len(group_metadata), 1)
        # self.assertEqual(
        #     group_metadata[0]["add"]["ideas"]["dataset"]["group_type"],
        #     "tool_output",
        # )

        # # validate series metadata
        # self.assertEqual(series_metadata, [])

        # # validate file metadata
        # self.assertEqual(len(file_metadata), 2)
        # for f in file_metadata:
        #     self.assertEqual(list(f["add"].keys()), ["ideas"])
        #     self.assertEqual(
        #         f["add"]["ideas"]["timingInfo"]["sampling_rate"], 10
        #     )

    def test_peri_event_workflow_plot_limits(self):
        # ensure that data previews are properly generated when either user-specified
        # or automatic plot limits are applied

        for (
            population_activity_plot_limits,
            activity_heatmap_color_limits,
            activity_by_modulation_plot_limits,
        ) in [
            ("-0.1,0.25", "-0.3,0.5", "-0.12,0.24"),
            ("auto", "auto", "auto"),
        ]:
            input_parameters = {
                "event_types": ["event_type_1"],
                "visual_window": {"pre": -2, "post": 2},
                "statistical_window": {"pre": [-1, 0], "post": [0, 1]},
                "num_shuffles": 50,
                "significance_threshold": 0.05,
                "seed": 0,
                "population_activity_plot_limits": population_activity_plot_limits,
                "activity_heatmap_color_limits": activity_heatmap_color_limits,
                "activity_by_modulation_plot_limits": activity_by_modulation_plot_limits,
            }

            # execute workflow
            run_peri_event_workflow(
                input_cellset_files=self.input_cellset_isxd_files,
                input_events_h5_file=self.input_events_h5_file,
                event_type=input_parameters["event_types"][0],
                visual_window_pre=input_parameters["visual_window"]["pre"],
                visual_window_post=input_parameters["visual_window"]["post"],
                statistical_window_pre_start=input_parameters[
                    "statistical_window"
                ]["pre"][0],
                statistical_window_pre_end=input_parameters[
                    "statistical_window"
                ]["pre"][1],
                statistical_window_post_start=input_parameters[
                    "statistical_window"
                ]["post"][0],
                statistical_window_post_end=input_parameters[
                    "statistical_window"
                ]["post"][1],
                num_shuffles=input_parameters["num_shuffles"],
                significance_threshold=input_parameters[
                    "significance_threshold"
                ],
                seed=input_parameters["seed"],
                population_activity_plot_limits=input_parameters[
                    "population_activity_plot_limits"
                ],
                activity_heatmap_color_limits=input_parameters[
                    "activity_heatmap_color_limits"
                ],
                activity_by_modulation_plot_limits=input_parameters[
                    "activity_by_modulation_plot_limits"
                ],
                output_dir=self.output_dir,
            )

            # validate existence of output preview files
            event_type = input_parameters["event_types"][0]
            output_dir = os.path.join(
                self.output_dir, "event_type_" + event_type
            )

            # define basename for output files
            population_activity_basename = os.path.join(
                output_dir, "event_aligned_population_activity"
            )
            modulation_basename = os.path.join(
                output_dir, "event_aligned_activity_by_modulation"
            )
            heatmap_basename = os.path.join(
                output_dir, "event_aligned_single_cell_activity_heatmap"
            )
            cell_map_basename = os.path.join(output_dir, "cell_map")

            # ensure PREVIEW files exist
            population_activity_preview_filename = (
                population_activity_basename + ".preview.svg"
            )
            modulation_preview_filename = modulation_basename + ".preview.svg"
            heatmap_preview_filename = heatmap_basename + ".preview.svg"
            cell_map_preview_filename = cell_map_basename + ".preview.svg"
            for f in [
                population_activity_preview_filename,
                modulation_preview_filename,
                heatmap_preview_filename,
                cell_map_preview_filename,
            ]:
                self.assertTrue(os.path.exists(f))

            # clean up output dir
            for f in os.listdir(output_dir):
                os.remove(os.path.join(output_dir, f))

    def test_nanmean_iterative(self):
        event_windows = np.zeros((2, 3, 4))
        cell_indices = [0, 2]
        expected = np.zeros((2, 3))
        actual = nanmean_iterative(event_windows, cell_indices)
        self.assertTrue(np.equal(actual, expected).all())

    def test_nanmean_iterative_zeros(self):
        event_windows = np.zeros((2, 3, 4))
        event_windows[:, :, 1] = 2 * np.ones((2, 3))
        event_windows[:, :, 3] = 4 * np.ones((2, 3))
        cell_indices = [1, 3]
        expected = 3 * np.ones((2, 3))
        actual = nanmean_iterative(event_windows, cell_indices)
        self.assertTrue(np.equal(actual, expected).all())

    def test_nanmean_iterative_single_value(self):
        event_windows = np.ones((2, 3, 4))
        cell_indices = [1, 3]
        expected = np.ones((2, 3))
        actual = nanmean_iterative(event_windows, cell_indices)
        self.assertTrue(np.equal(actual, expected).all())

    def test_nanmean_iterative_single_cell(self):
        event_windows = np.zeros((2, 3, 4))
        event_windows[:, :, 2] = np.ones((2, 3))
        cell_indices = [2]
        expected = np.ones((2, 3))
        actual = nanmean_iterative(event_windows, cell_indices)
        self.assertTrue(np.equal(actual, expected).all())

    def test_nanmean_iterative_empty_list_of_cells(self):
        event_windows = np.zeros((2, 3, 4))
        cell_indices = []
        self.assertRaisesRegex(
            IdeasError,
            "Cannot compute the mean of 0 elements",
            nanmean_iterative,
            event_windows,
            cell_indices,
        )

    def test_nanmean_iterative_empty_input_array(self):
        event_windows = np.array([[[]]])
        cell_indices = [0, 2]
        self.assertRaisesRegex(
            IdeasError,
            "Cannot compute the mean of 0 elements",
            nanmean_iterative,
            event_windows,
            cell_indices,
        )

    def test_peri_event_workflow_accepted_cells_only(self):
        """All cells are marked as accepted"""
        # define input parameters
        input_cellset_isxd_files = [
            os.path.join(self.input_dir, "cellset_accepted_cells_only.isxd")
        ]
        input_parameters = {
            "event_types": ["event_type_1"],
            "visual_window": {"pre": -2, "post": 2},
            "statistical_window": {"pre": [-1, 0], "post": [0, 1]},
            "num_shuffles": 50,
            "significance_threshold": 0.05,
            "seed": 0,
        }
        event_type = input_parameters["event_types"][0]

        # execute workflow
        run_peri_event_workflow(
            input_cellset_files=input_cellset_isxd_files,
            input_events_h5_file=self.input_events_h5_file,
            event_type=event_type,
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
            output_dir=self.output_dir,
        )

        # validate existence of output files
        self.assertEqual(
            sorted(os.listdir(self.output_dir)),
            sorted(
                [
                    # "output_manifest.json",
                    # "output_metadata.json",
                    f"event_type_{event_type}",
                ]
            ),
        )

        event_type_output_dir = os.path.join(
            self.output_dir, f"event_type_{event_type}"
        )
        actual_files = os.listdir(event_type_output_dir)
        expected_files = [
            "cell_map.preview.svg",
            "event_aligned_activity.STATISTICS.csv",
            "event_aligned_activity.TRACES.csv",
            "event_aligned_activity_by_modulation.preview.svg",
            "event_aligned_population_activity.preview.svg",
            "event_aligned_single_cell_activity_heatmap.preview.svg",
        ]
        self.assertEqual(sorted(actual_files), sorted(expected_files))

    def test_peri_event_workflow_single_accepted_cell(self):
        """A single cell is marked as accepted. All other cells are undecided or rejected."""
        # define input parameters
        input_cellset_isxd_files = [
            os.path.join(self.input_dir, "cellset_single_accepted_cell.isxd")
        ]
        input_parameters = {
            "event_types": ["event_type_1"],
            "visual_window": {"pre": -2, "post": 2},
            "statistical_window": {"pre": [-1, 0], "post": [0, 1]},
            "num_shuffles": 50,
            "significance_threshold": 0.05,
            "seed": 0,
        }
        event_type = input_parameters["event_types"][0]

        # execute workflow
        run_peri_event_workflow(
            input_cellset_files=input_cellset_isxd_files,
            input_events_h5_file=self.input_events_h5_file,
            event_type=event_type,
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
            output_dir=self.output_dir,
        )

        # validate existence of output files
        self.assertEqual(
            sorted(os.listdir(self.output_dir)),
            sorted(
                [
                    # "output_manifest.json",
                    # "output_metadata.json",
                    f"event_type_{event_type}",
                ]
            ),
        )

        event_type_output_dir = os.path.join(
            self.output_dir, f"event_type_{event_type}"
        )
        actual_files = os.listdir(event_type_output_dir)
        expected_files = [
            "cell_map.preview.svg",
            "event_aligned_activity.STATISTICS.csv",
            "event_aligned_activity.TRACES.csv",
            "event_aligned_activity_by_modulation.preview.svg",
            "event_aligned_population_activity.preview.svg",
            "event_aligned_single_cell_activity_heatmap.preview.svg",
        ]
        self.assertEqual(sorted(actual_files), sorted(expected_files))

    def test_peri_event_workflow_mixture_of_cell_statuses(self):
        """Cell set with cells in all statuses (accepted, rejected, undecided"""
        # define input parameters
        input_cellset_isxd_files = [
            os.path.join(
                self.input_dir,
                "cellset_accepted_undecided_and_rejected_cells.isxd",
            )
        ]
        input_parameters = {
            "event_types": ["event_type_1"],
            "visual_window": {"pre": -2, "post": 2},
            "statistical_window": {"pre": [-1, 0], "post": [0, 1]},
            "num_shuffles": 50,
            "significance_threshold": 0.05,
            "seed": 0,
        }
        event_type = input_parameters["event_types"][0]

        # execute workflow
        run_peri_event_workflow(
            input_cellset_files=input_cellset_isxd_files,
            input_events_h5_file=self.input_events_h5_file,
            event_type=event_type,
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
            output_dir=self.output_dir,
        )

        # validate existence of output files
        self.assertEqual(
            sorted(os.listdir(self.output_dir)),
            sorted(
                [
                    # "output_manifest.json",
                    # "output_metadata.json",
                    f"event_type_{event_type}",
                ]
            ),
        )

        event_type_output_dir = os.path.join(
            self.output_dir, f"event_type_{event_type}"
        )
        actual_files = os.listdir(event_type_output_dir)
        expected_files = [
            "cell_map.preview.svg",
            "event_aligned_activity.STATISTICS.csv",
            "event_aligned_activity.TRACES.csv",
            "event_aligned_activity_by_modulation.preview.svg",
            "event_aligned_population_activity.preview.svg",
            "event_aligned_single_cell_activity_heatmap.preview.svg",
        ]
        self.assertEqual(sorted(actual_files), sorted(expected_files))

    def test_peri_event_workflow_rejected_cells_only(self):
        """All cells are marked as rejected"""
        # define input parameters
        input_cellset_isxd_files = [
            os.path.join(self.input_dir, "cellset_rejected_cells_only.isxd")
        ]
        input_parameters = {
            "event_types": ["event_type_1"],
            "visual_window": {"pre": -2, "post": 2},
            "statistical_window": {"pre": [-1, 0], "post": [0, 1]},
            "num_shuffles": 50,
            "significance_threshold": 0.05,
            "seed": 0,
        }
        event_type = input_parameters["event_types"][0]

        self.assertRaisesRegex(
            IdeasError,
            "There are no accepted or undecided cells to process",
            run_peri_event_workflow,
            input_cellset_files=input_cellset_isxd_files,
            input_events_h5_file=self.input_events_h5_file,
            event_type=event_type,
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
            output_dir=self.output_dir,
        )

    def test_peri_event_workflow_undecided_cells_only(self):
        """All cells are marked as undecided"""
        # define input parameters
        input_cellset_isxd_files = [
            os.path.join(self.input_dir, "cellset_undecided_cells_only.isxd")
        ]
        input_parameters = {
            "event_types": ["event_type_1"],
            "visual_window": {"pre": -2, "post": 2},
            "statistical_window": {"pre": [-1, 0], "post": [0, 1]},
            "num_shuffles": 50,
            "significance_threshold": 0.05,
            "seed": 0,
        }
        event_type = input_parameters["event_types"][0]

        # execute workflow
        run_peri_event_workflow(
            input_cellset_files=input_cellset_isxd_files,
            input_events_h5_file=self.input_events_h5_file,
            event_type=event_type,
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
            output_dir=self.output_dir,
        )

        # validate existence of output files
        self.assertEqual(
            sorted(os.listdir(self.output_dir)),
            sorted(
                [
                    # "output_manifest.json",
                    # "output_metadata.json",
                    f"event_type_{event_type}",
                ]
            ),
        )

        event_type_output_dir = os.path.join(
            self.output_dir, f"event_type_{event_type}"
        )
        actual_files = os.listdir(event_type_output_dir)
        expected_files = [
            "cell_map.preview.svg",
            "event_aligned_activity.STATISTICS.csv",
            "event_aligned_activity.TRACES.csv",
            "event_aligned_activity_by_modulation.preview.svg",
            "event_aligned_population_activity.preview.svg",
            "event_aligned_single_cell_activity_heatmap.preview.svg",
        ]
        self.assertEqual(sorted(actual_files), sorted(expected_files))
