
VERSION=latest
DEFAULT_IMAGE_TAG=inscopix/neuro-stats:${VERSION}
PLATFORM=linux/amd64

IMAGE_TAG := $(if $(IMAGE_TAG),$(IMAGE_TAG),$(DEFAULT_IMAGE_TAG))

.PHONY: build

.DEFAULT_GOAL := build

build:
	@echo "Building docker image..."
	docker build . -t $(IMAGE_TAG) \
		--platform ${PLATFORM}

test: build 
	@echo "Running tests..."
	docker run \
		--platform ${PLATFORM} \
		--rm \
		-v $(PWD):/neural_circuit_analysis \
		-w /neural_circuit_analysis \
		${IMAGE_TAG} \
		pytest ${TEST_ARGS}
