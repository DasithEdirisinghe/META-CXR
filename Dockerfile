FROM meta-cxr:1.0.0

WORKDIR /workspace/META-CXR

ENTRYPOINT ["/bin/bash", "inference.sh"]