#!/bin/bash

CONTAINER_NAME="meta-cxr-container"
IMAGE_NAME="meta-cxr:2.0.0"
WORKDIR="$(pwd)"

# Check if container exists
if [ "$(docker ps -a -q -f name=^/${CONTAINER_NAME}$)" ]; then
    echo "Stopping and removing existing container: ${CONTAINER_NAME}"
    docker stop ${CONTAINER_NAME}
    docker rm ${CONTAINER_NAME}
fi

echo "Starting new container: ${CONTAINER_NAME}"

docker run -it \
  --restart=unless-stopped \
  --privileged \
  --gpus all \
  -v "${WORKDIR}:/workspace/META-CXR" \
  --name "${CONTAINER_NAME}" \
  "${IMAGE_NAME}"


# docker run -it --restart=unless-stopped --privileged --gpus all -v $(pwd):/workspace/META-CXR --name meta-cxr-container --entrypoint /bin/bash meta-cxr:2.0.0 