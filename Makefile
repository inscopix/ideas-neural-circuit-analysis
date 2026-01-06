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

# Update the tool specs whenever a new version of a container image is created
TOOL_SPECS=${shell ls -d .ideas/*/tool_spec.json}

.DEFAULT_GOAL := build

clean:
	@echo "Cleaning up"
	-docker rmi ${IMAGE_TAG}

# Builds docker image
# Installs necessary software dependencies for source code
build:
	docker build . -t $(LATEST_IMAGE_TAG) \
		--platform ${PLATFORM} \
		--target ${TARGET}
	docker tag ${LATEST_IMAGE_TAG} ${IMAGE_TAG}
	@$(foreach f, $(TOOL_SPECS), jq --indent 4 '.container_image.label = "${LABEL}"' $(f) > tmp.json && mv tmp.json ${f};)\

# Builds docker image
# Copies the source code into the docker container for isolated testing environment
build-test: TARGET=test
build-test: IMAGE_TAG=${IMAGE_REPO}/${IMAGE_NAME}:${LABEL}-test
build-test: LATEST_IMAGE_TAG=${IMAGE_REPO}/${IMAGE_NAME}:latest-test
build-test: build

# Runs unit tests in docker image
# Used in automated pr checks on github
test: clean build-test
	@echo "Running tests..."
	docker run \
		--platform ${PLATFORM} \
		--rm \
		${IMAGE_TAG} \
		python -m pytest ${TEST_ARGS}

# Applies linter on source code
# The source code is volume mounted instead of copied into the docker image
# so that the formatting changes are made on the local files instead of just in the docker image
ruff: build
	@echo "Running tests..."
	docker run \
		--platform ${PLATFORM} \
		--rm \
		-v $(PWD):/ideas/${REPO_NAME} \
		${IMAGE_TAG} \
		bash -c "python -m ruff format /ideas/${REPO_NAME} $(ARGS) && python -m ruff check --fix /ideas/${REPO_NAME} $(ARGS)"

# Checks code formatting of source code
# Used in automated pr checks on github
# Does not actually apply any linter changes on the source code,
# only checks if all files are formatted correctly
ruff-check: build-test
	docker run \
		--platform ${PLATFORM} \
		--rm \
		${IMAGE_TAG} \
		bash -c "python -m ruff format --check /ideas $(ARGS) && python -m ruff check --no-fix /ideas $(ARGS)"

lint: ruff