.PHONY:  clean build test

IMAGE_REPO=platform
IMAGE_NAME=neuro-stats

# Update this value anytime you're publishing a new version of this image to IDEAS
ifndef LABEL
	LABEL=0.0.1
endif

IMAGE_TAG=${IMAGE_REPO}/${IMAGE_NAME}:${LABEL}
PLATFORM=linux/amd64
ifndef TARGET
	TARGET=base
endif

.DEFAULT_GOAL := build

clean:
	@echo "Cleaning up"
	-docker rmi ${IMAGE_TAG}

build: 
	docker build . -t $(IMAGE_TAG) \
		--platform ${PLATFORM} \
		--target ${TARGET}

test: TARGET=test
test: IMAGE_TAG=${IMAGE_REPO}/${IMAGE_NAME}:${LABEL}-test
test: clean build 
	@echo "Running tests..."
	docker run \
		--platform ${PLATFORM} \
		--rm \
		${IMAGE_TAG} \
		pytest ${TEST_ARGS}
