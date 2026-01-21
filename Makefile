.PHONY:  clean build test venv

REPO_NAME=neural-circuit-analysis
IMAGE_REPO=platform
IMAGE_NAME=neural-analysis
LABEL=$(shell cat .ideas/images_spec.json | jq -r ".[0].label")
IMAGE_TAG=${IMAGE_REPO}/${IMAGE_NAME}:${LABEL}
LATEST_IMAGE_TAG=${IMAGE_REPO}/${IMAGE_NAME}:latest
PLATFORM=linux/amd64
ifndef TARGET
	TARGET=base
endif

# Define envs for virtualenv
ROOT_DIR := $(shell dirname $(shell readlink -f $(firstword $(MAKEFILE_LIST))))
VENV = $(ROOT_DIR)/venv

# Detect OS to know how to call python in venv
ifeq ($(OS), Windows_NT)
	PYTHON = $(VENV)/Scripts/python
	PRECOMMIT = $(VENV)/Scripts/pre-commit
else
	PYTHON = $(VENV)/bin/python
	PRECOMMIT = $(VENV)/bin/pre-commit
endif

# Update the tool specs whenever a new version of a container image is created
TOOL_SPECS=${shell ls -d .ideas/*/tool_spec.json}

.DEFAULT_GOAL := build

clean:
	@echo "Cleaning up"
	-docker rmi ${IMAGE_TAG}
	-docker rmi ${IMAGE_TAG}-test

clean-venv:
	rm -rf venv .venv

venv: venv/touchfile

venv/touchfile: pyproject.toml uv.lock
	test -d $(VENV) || uv venv && ln -sf .venv $(VENV)
	uv sync --no-install-project --only-group dev
	touch $(VENV)/touchfile

set-hooks: venv .pre-commit-config.yaml
	@echo "Installing pre-commit hooks"
	$(PRECOMMIT) install

setup: venv set-hooks

# Builds docker image
# Installs necessary software dependencies for source code
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
	$(PYTHON) -m ruff format . $(ARGS)
	$(PYTHON) -m ruff check --fix . $(ARGS)

# Checks code formatting of source code
# Used in automated pr checks on github
# Does not actually apply any linter changes on the source code,
ruff-check: venv
	@echo "Running lint..."
	$(PYTHON) -m ruff format --check . $(ARGS)
	$(PYTHON) -m ruff check --no-fix . $(ARGS)

# Run a tool in the repo
# Specify the tool key to run
run: build
	ideas tools run $(tool) -s -c -n

run-all: build
	@$(foreach f, $(shell ls -d .ideas/*), ideas tools run -s -c -n $(shell basename $(f));)
