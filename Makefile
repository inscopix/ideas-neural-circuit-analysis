.PHONY:  clean build test

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
VENV = venv

ifndef PYTHON_VERSION
	PYTHON_VERSION=python3.13
endif

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
	-rm -rf venv

venv: pyproject.toml
	@echo "Creating virtualenv and installing dependencies"
	test -d venv || $(PYTHON_VERSION) -m venv venv
	$(PYTHON) -m pip install pip --upgrade
	$(PYTHON) -m pip install '.[dev]'
	# Let make know the venv is up-to-date
	touch venv

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
test: clean build
	@echo "Running tests..."
	docker run \
		--platform ${PLATFORM} \
		--rm \
		${IMAGE_TAG} \
		python -m pytest ${TEST_ARGS}

# Applies linter on source code
ruff: venv
	$(PYTHON) -m ruff format . $(ARGS)
	$(PYTHON) -m ruff check --fix . $(ARGS)

# Checks code formatting of source code
# Used in automated pr checks on github
# Does not actually apply any linter changes on the source code,
ruff-check: venv
	$(PYTHON) -m ruff format --check . $(ARGS)
	$(PYTHON) -m ruff check --no-fix . $(ARGS)

lint: ruff