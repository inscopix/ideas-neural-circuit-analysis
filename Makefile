.PHONY:  clean clean-venv venv set-hooks setup build test ruff ruff-check run run-all

IMAGE_REPO=platform
IMAGE_NAME=neural-analysis
LABEL=$(shell cat .ideas/images_spec.json | jq -r ".[0].label")
IMAGE_TAG=${IMAGE_REPO}/${IMAGE_NAME}:${LABEL}
LATEST_IMAGE_TAG=${IMAGE_REPO}/${IMAGE_NAME}:latest
PLATFORM=linux/amd64
ifndef TARGET
	TARGET=final
endif

# Define envs for virtualenv
ROOT_DIR := $(shell dirname $(shell readlink -f $(firstword $(MAKEFILE_LIST))))
VENV = $(ROOT_DIR)/.venv

# IDEAS CLI is used to execute tools locally
# It's installed in the project venv, but can be overwritten to a customized local version instead.
ifndef IDEAS_CLI
	IDEAS_CLI=${VENV}/bin/ideas
endif

# Update the tool specs whenever a new version of a container image is created
TOOL_SPECS=${shell ls -d .ideas/*/tool_spec.json}

.DEFAULT_GOAL := build

clean:
	@echo "Cleaning up"
	-docker rmi ${IMAGE_TAG}
	-docker rmi ${IMAGE_TAG}-test

clean-venv:
	rm -rf $(VENV)

validate-ideas-cli:
	@test -f ${IDEAS_CLI} || { echo "Error: IDEAS CLI not found. Ensure you have run 'make venv', or override and use your own install of the cli by specifying a value for IDEAS_CLI when executing make commands"; exit 1; }

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
# Installs necessary software dependencies for source code
build:
	docker build . -t $(LATEST_IMAGE_TAG) \
		--platform ${PLATFORM} \
		--target ${TARGET}
	docker tag ${LATEST_IMAGE_TAG} ${IMAGE_TAG}
	@$(foreach f, $(TOOL_SPECS), jq --indent 4 '.container_image.label = "${LABEL}"' $(f) > tmp.json && mv tmp.json ${f};)\

# Builds docker image checking security vulnerabilities
security:
	docker build . -t $(LATEST_IMAGE_TAG) \
		--platform ${PLATFORM}

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
		python3 -m pytest ${TEST_ARGS}

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
run: build validate-ideas-cli
	${IDEAS_CLI} tools run $(tool) -s -c -n

run-all: build validate-ideas-cli
	@set -e; \
	for f in .ideas/*/; do \
		${IDEAS_CLI} tools run -s -c -n "$$(basename "$$f")"; \
	done

run-all-fresh: clean-venv venv run-all
