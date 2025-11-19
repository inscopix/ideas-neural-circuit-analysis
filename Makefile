.PHONY:  clean build test

IMAGE_REPO=platform
IMAGE_NAME=neuro-stats
LABEL=$(shell cat .ideas/images_spec.json | jq -r ".[0].label")
IMAGE_TAG=${IMAGE_REPO}/${IMAGE_NAME}:${LABEL}
LATEST_IMAGE_TAG=${IMAGE_REPO}/${IMAGE_NAME}:latest
PLATFORM=linux/amd64
ifndef TARGET
	TARGET=base
endif

# Update the tool specs whenever a new version of a container imagee is created
TOOL_SPECS=${shell ls -d .ideas/*/tool_spec.json}

.DEFAULT_GOAL := build

clean:
	@echo "Cleaning up"
	-docker rmi ${IMAGE_TAG}

build:
	docker build . -t $(LATEST_IMAGE_TAG) \
		--platform ${PLATFORM} \
		--target ${TARGET}
	docker tag ${LATEST_IMAGE_TAG} ${IMAGE_TAG}
	@$(foreach f, $(TOOL_SPECS), jq --indent 4 '.container_image.label = "${LABEL}"' $(f) > tmp.json && mv tmp.json ${f};)

test: TARGET=test
test: IMAGE_TAG=${IMAGE_REPO}/${IMAGE_NAME}:${LABEL}-test
test: LATEST_IMAGE_TAG=${IMAGE_REPO}/${IMAGE_NAME}:latest-test
test: clean build 
	@echo "Running tests..."
	docker run \
		--platform ${PLATFORM} \
		--rm \
		${IMAGE_TAG} \
		pytest ${TEST_ARGS}
