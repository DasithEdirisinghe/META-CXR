#!/bin/bash

IMAGE_NAME="meta-cxr"
IMAGE_TAG="1.0.0"
IMAGE_FULL="${IMAGE_NAME}:${IMAGE_TAG}"
DOCKERHUB_IMAGE="dasithdev/${IMAGE_FULL}"

# Check if local image exists
if [[ "$(docker images -q ${IMAGE_FULL} 2> /dev/null)" == "" ]]; then
    echo "Image ${IMAGE_FULL} not found locally. Pulling from Docker Hub..."
    docker pull ${DOCKERHUB_IMAGE}
    docker tag ${DOCKERHUB_IMAGE} ${IMAGE_FULL}
else
    echo "Image ${IMAGE_FULL} already exists locally."
fi

# Build new Docker image on top
docker build -t meta-cxr:2.0.0 -f Dockerfile .