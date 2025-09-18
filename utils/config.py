# metadata keys
IDEAS_METADATA_KEY = "ideas"

# GPIO keys
GPIO_SIGNALS_H5_KEY = "gpio_signals"
GPIO_OFFSETS_H5_KEY = "offsets"
GPIO_AMPLITUDES_H5_KEY = "amplitudes"

# output file suffixes & extensions
OUTPUT_PREVIEW_JSON_FILE_EXTENSION = ".preview.json"
OUTPUT_PREVIEW_SVG_FILE_EXTENSION = ".preview.svg"
OUTPUT_METADATA_JSON_FILE_EXTENSION = ".metadata.json"
OUTPUT_DATA_JSON_FILE_EXTENSION = ".data.json"

# peri-event workflow plots
PLOT_TITLE_FONT_SIZE = 14
PLOT_LABEL_FONT_SIZE = 14
PLOT_SHADING_ALPHA = 0.2
PLOT_UP_MODULATED_COLOR = "green"
PLOT_DOWN_MODULATED_COLOR = "blue"
PLOT_NON_MODULATED_COLOR = "black"
PLOT_EVENT_REF_LINE_COLOR = "red"

# log files
LOG_EXIT_STATUS_FILE = "exit_status.txt"
LINE_BY_LINE_PROFILER_LOG_FILENAME = "line_by_line_profiler_log.txt"
TOOLS_PROFILER_LOG_FILENAME = "tools_profiler.txt"
TOOLS_PROFILER_CSV_FILENAME = "tools_profiler.csv"
TOOLBOX_INFO_FILE = "toolbox_info.json"
RESOURCE_ESTIMATOR_LOG_FILENAME = "tools_max_resources.txt"
RESOURCE_ESTIMATOR_CSV_FILENAME = "tools_resources.csv"
SINGLE_TOOL_PROFILING_CSV_FILENAME = "profiling_results.csv"

# output manifest
OUTPUT_MANIFEST_FILE = "output_manifest.json"
OUTPUT_METADATA_MANIFEST_FILE = "output_metadata.json"

# schemas
OUTPUT_MANIFEST_SCHEMA_FILE = (
    "/ideas/toolbox/tests/schemas/output_manifest_schema.json"
)
OUTPUT_METADATA_SCHEMA_FILE = (
    "/ideas/toolbox/tests/schemas/output_metadata_schema.json"
)
TOOLBOX_SPEC_SCHEMA_FILE = (
    "/ideas/toolbox/tests/schemas/toolbox_spec_schema.json"
)
IDPS_METADATA_SCHEMA_FILE = (
    "/ideas/toolbox/tests/schemas/idps_metadata_schema.json"
)

# tool versions file
TOOL_VERSIONS_FILE = "/ideas/toolbox/tool_versions.json"
TOOLBOX_REPO_NAME = "ideas-toolbox-standard-python"

# path to the directories: commands and their inputs
COMMAND_DIR_PATH = "/ideas/commands/"
COMMAND_INPUTS_DIR_PATH = "/ideas/toolbox/tests/command_inputs/"

# DEFAULT RESOURCE USAGE
# Storage, RAM. max file size values in MB
DEFAULT_CPU = 4
DEFAULT_GPU = 0
DEFAULT_RAM = 4096
DEFAULT_STORAGE = 8192
DEFAULT_MAX_FILE_SIZE = 1024

# AWS environment variables
AWS_ENV_CODEBUILD_ID = "CODEBUILD_BUILD_ID"

# Files for formula estimation
FORMULA_FILE = "/ideas/resources/tools_resources_formula.json"
