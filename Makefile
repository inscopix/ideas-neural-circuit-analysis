
VERSION=latest

IMAGE_TAG=neuro-stats:${VERSION}
PLATFORM=linux/amd64

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
