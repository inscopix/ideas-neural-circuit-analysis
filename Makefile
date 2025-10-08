PLATFORM=linux/amd64

# Label may be specified in codebuild pipeline
# Locally, use default "latest"
ifndef LABEL
	LABEL=latest
endif

IMAGE_TAG := platform/neuro-stats:${LABEL}

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
