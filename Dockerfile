# Build sandbox for the untrusted Strata checkout: CUDA 13.0 toolkit (the one Strata releases with) plus compilers.
# nvidia/cuda:13.0.0-devel-ubuntu24.04, pinned by digest (multi-arch index; arm64 is used on Grace Hopper).
# full registry name: podman refuses short names without a TTY
FROM docker.io/nvidia/cuda@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git libatomic1 libgomp1 libnuma1 python3 python3-pip python3-venv unzip \
    && rm -rf /var/lib/apt/lists/*
