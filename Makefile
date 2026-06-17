.PHONY:  clean clean-venv venv set-hooks setup build test ruff ruff-check run run-all

IMAGE_REPO=platform
IMAGE_NAME=neural-analysis
LABEL=$(shell cat .ideas/images_spec.json | jq -r ".[0].label")
IMAGE_TAG=${IMAGE_REPO}/${IMAGE_NAME}:${LABEL}
LATEST_IMAGE_TAG=${IMAGE_REPO}/${IMAGE_NAME}:latest
PLATFORM=linux/amd64
ifndef TARGET
	TARGET=base
endif
RUN_TOOL := $(strip $(if $(TOOL),$(TOOL),$(tool)))
IDEAS_RUN_FLAGS ?= -s -c -n
LOCAL_DATA_DIR ?=

# Define envs for virtualenv
ROOT_DIR := $(shell dirname $(shell readlink -f $(firstword $(MAKEFILE_LIST))))
REPO_NAME := $(notdir $(ROOT_DIR))
CONTAINER_CODE_DIR := /tmp/$(REPO_NAME)
VENV = $(ROOT_DIR)/.venv

# Update the tool specs whenever a new version of a container image is created
TOOL_SPECS=${shell ls -d .ideas/*/tool_spec.json}

.DEFAULT_GOAL := build

clean:
	@echo "Cleaning up"
	-docker rmi ${IMAGE_TAG}
	-docker rmi ${IMAGE_TAG}-test

clean-venv:
	rm -rf $(VENV)

venv: .venv/touchfile

.venv/touchfile: pyproject.toml
	test -d $(VENV) || uv venv
	uv sync --no-install-project --only-group dev
	touch $(VENV)/touchfile

set-hooks: venv .pre-commit-config.yaml
	@echo "Installing pre-commit hooks"
	uv run pre-commit install

setup: venv set-hooks

# Builds docker image
build:
	docker build . -t $(LATEST_IMAGE_TAG) \
		--platform ${PLATFORM} \
		--target ${TARGET}
	docker tag ${LATEST_IMAGE_TAG} ${IMAGE_TAG}
	@$(foreach f, $(TOOL_SPECS), jq --indent 4 '.container_image.label = "${LABEL}"' $(f) > tmp.json && mv tmp.json ${f};)\

# Runs unit tests in docker image
# Used in automated pr checks on github
test: TARGET=test
test: IMAGE_TAG=${IMAGE_REPO}/${IMAGE_NAME}:${LABEL}-test
test: LATEST_IMAGE_TAG=${IMAGE_REPO}/${IMAGE_NAME}:latest-test
test: build
	@echo "Running tests..."
	docker run \
		--platform ${PLATFORM} \
		--rm \
		${IMAGE_TAG} \
		python -m pytest ${TEST_ARGS}

# Applies linter on source code
ruff: venv
	@echo "Running ruff..."
	uv run ruff format . $(ARGS)
	uv run ruff check --fix . $(ARGS)

# Checks code formatting of source code
# Used in automated pr checks on github
# Does not actually apply any linter changes on the source code,
ruff-check: venv
	@echo "Running lint..."
	uv run ruff format --check . $(ARGS)
	uv run ruff check --no-fix . $(ARGS)

# Run a tool in the repo
# Specify the tool key to run
# Optional: override flags, e.g. IDEAS_RUN_FLAGS="-c -n"
# Optional: set LOCAL_DATA_DIR to remap data/... paths (uses --inputs if provided)
run: build
	@if [ -z "$(RUN_TOOL)" ]; then \
		echo "Tool key required."; \
		echo "Usage: make run tool=<tool-key> (or TOOL=<tool-key>)"; \
		exit 2; \
	fi
	@if [ ! -d ".ideas/$(RUN_TOOL)" ]; then \
		echo "Unknown tool key: $(RUN_TOOL)"; \
		echo "Available tools:"; \
		ls -1 .ideas | sed 's/^/  - /'; \
		exit 2; \
	fi
	@TMP_INPUTS_FILE="/tmp/ideas-local-inputs.$(RUN_TOOL).json"; \
	RUN_INPUTS_ARG=""; \
	RUN_FLAGS_CLEAN="$(IDEAS_RUN_FLAGS)"; \
	if [ -n "$(LOCAL_DATA_DIR)" ]; then \
		SOURCE_INPUTS_FILE="$$(printf '%s\n' "$(IDEAS_RUN_FLAGS)" | awk '{for(i=1;i<=NF;i++){if($$i=="--inputs"){print $$(i+1); exit}}}')"; \
		if [ -z "$$SOURCE_INPUTS_FILE" ]; then \
			SOURCE_INPUTS_FILE=".ideas/$(RUN_TOOL)/inputs.json"; \
		fi; \
		if [ ! -f "$$SOURCE_INPUTS_FILE" ]; then \
			echo "Inputs file not found: $$SOURCE_INPUTS_FILE"; \
			exit 2; \
		fi; \
		RUN_FLAGS_CLEAN="$$(printf '%s\n' "$(IDEAS_RUN_FLAGS)" | awk 'BEGIN{sep=""; skip=0} {for(i=1;i<=NF;i++){if(skip){skip=0; continue} if($$i=="--inputs"){skip=1; continue} printf "%s%s", sep, $$i; sep=" "}} END{print ""}')"; \
		LOCAL_DATA_DIR_ABS="$$(cd "$(LOCAL_DATA_DIR)" 2>/dev/null && pwd)"; \
		if [ -z "$$LOCAL_DATA_DIR_ABS" ]; then \
			echo "LOCAL_DATA_DIR not found: $(LOCAL_DATA_DIR)"; \
			exit 2; \
		fi; \
		if [ "$$LOCAL_DATA_DIR_ABS" = "$(ROOT_DIR)" ]; then \
			CONTAINER_DATA_DIR="$(CONTAINER_CODE_DIR)"; \
		elif [ "$${LOCAL_DATA_DIR_ABS#$(ROOT_DIR)/}" != "$$LOCAL_DATA_DIR_ABS" ]; then \
			REL_DATA_DIR="$${LOCAL_DATA_DIR_ABS#$(ROOT_DIR)/}"; \
			CONTAINER_DATA_DIR="$(CONTAINER_CODE_DIR)/$$REL_DATA_DIR"; \
		else \
			echo "LOCAL_DATA_DIR must be inside the repository: $(ROOT_DIR)"; \
			exit 2; \
		fi; \
		jq --indent 4 --arg root "$$CONTAINER_DATA_DIR" \
			'walk(if type=="string" and startswith("data/") then ($$root + "/" + ltrimstr("data/")) else . end)' \
			"$$SOURCE_INPUTS_FILE" > "$$TMP_INPUTS_FILE"; \
		RUN_INPUTS_ARG="--inputs $$TMP_INPUTS_FILE"; \
	fi; \
	IDEAS_PYENV_VERSION="$$(if command -v pyenv >/dev/null 2>&1; then pyenv whence ideas 2>/dev/null | awk 'NR==1{print; exit}'; fi)"; \
	STATUS=0; \
	if [ -n "$$IDEAS_PYENV_VERSION" ] && PYENV_VERSION="$$IDEAS_PYENV_VERSION" ideas tools --help >/dev/null 2>&1; then \
		PYENV_VERSION="$$IDEAS_PYENV_VERSION" ideas tools run "$(RUN_TOOL)" $$RUN_INPUTS_ARG $$RUN_FLAGS_CLEAN; \
	elif command -v ideas >/dev/null 2>&1 && ideas tools --help >/dev/null 2>&1; then \
		ideas tools run "$(RUN_TOOL)" $$RUN_INPUTS_ARG $$RUN_FLAGS_CLEAN; \
	elif command -v uv >/dev/null 2>&1 && uv run --no-sync ideas tools --help >/dev/null 2>&1; then \
		uv run ideas tools run "$(RUN_TOOL)" $$RUN_INPUTS_ARG $$RUN_FLAGS_CLEAN; \
	else \
		echo "No IDEAS CLI with 'tools' subcommand found."; \
		echo "Install ideas-python in your active env (e.g. 'uv pip install -U ideas-python')"; \
		STATUS=2; \
	fi || STATUS=$$?; \
	rm -f "$$TMP_INPUTS_FILE"; \
	exit $$STATUS

run-all: build
	@IDEAS_PYENV_VERSION="$$(if command -v pyenv >/dev/null 2>&1; then pyenv whence ideas 2>/dev/null | awk 'NR==1{print; exit}'; fi)"; \
	for f in .ideas/*/; do \
		tool="$$(basename "$$f")"; \
		if command -v uv >/dev/null 2>&1; then \
			if [ -n "$$IDEAS_PYENV_VERSION" ]; then \
				PYENV_VERSION="$$IDEAS_PYENV_VERSION" uv run ideas tools run "$$tool" $(IDEAS_RUN_FLAGS); \
			else \
				uv run ideas tools run "$$tool" $(IDEAS_RUN_FLAGS); \
			fi; \
		elif [ -n "$$IDEAS_PYENV_VERSION" ]; then \
			PYENV_VERSION="$$IDEAS_PYENV_VERSION" ideas tools run "$$tool" $(IDEAS_RUN_FLAGS); \
		else \
			ideas tools run "$$tool" $(IDEAS_RUN_FLAGS); \
		fi; \
	done
