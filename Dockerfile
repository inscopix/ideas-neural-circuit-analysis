# Create base image to run analysis
FROM public.ecr.aws/docker/library/python:3.13.12 AS base

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONFAULTHANDLER=1
ENV DEBIAN_FRONTEND=noninteractive

# Arguments for python installation
ARG PYTHON=python
ARG VENV=venv
ARG PYTHON_VENV=/ideas/${VENV}/bin/python

# Create ideas user
# This is no longer necessary to do, but good practice anyways
RUN addgroup ideas \
    && adduser --disabled-password --home /ideas --ingroup ideas ideas

# Create ideas home dir
WORKDIR /ideas

# ========================== Apt Dependency Installation ===========================
RUN apt-get -y update \
    && apt-get -y upgrade \
    && apt-get install -y libgl1 --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Create a venv with uv to install python dependencies
# This can be done globally, but using venv is best practice

# ========================== Python Dependency Installation with uv ===========================

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy
# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
# Ensure installed tools can be executed out of the box
ENV UV_TOOL_BIN_DIR=/usr/local/bin
# Install venv into in the ideas user's home
ENV UV_PROJECT_ENVIRONMENT=/ideas/venv VIRTUAL_ENV=/ideas/venv
# Never download python, we use upstream python from python docker image
ENV UV_NO_MANAGED_PYTHON=1 UV_PYTHON_DOWNLOADS=never
# Strictly use frozen dependencies, and require cryptographic verification of each package
ENV UV_FROZEN=1 UV_REQUIRE_HASHES=1 UV_VERIFY_HASHES=1
ENV UV_CACHE_DIR=/tmp/.cache/uv

RUN --mount=from=ghcr.io/astral-sh/uv:0.10.11,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/tmp/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --group analysis

USER ideas

# Add venv bin to path
ENV PATH="/ideas/${VENV}/bin:${PATH}"

CMD ["/bin/bash"]

# Create image for testing which copies tool code and test data to
# docker image in order to facilitate unit testing in an isolated environment.
# This can also be acheived with volume mounts, but that can clutter up
# your local folder with files generated during testing.
FROM base AS test

USER root

COPY --chown=ideas ./ /ideas

RUN --mount=from=ghcr.io/astral-sh/uv:0.10.11,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/tmp/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --group analysis --group test

USER ideas

# Vulnerability scanning stage using Trivy
# Copies the runtime filesystem to a subdirectory to avoid overwrite conflicts with the trivy base image
FROM base AS scanner

USER root

COPY --from=aquasec/trivy:0.69.3 /usr/local/bin/trivy /usr/local/bin/trivy

RUN trivy rootfs --no-progress --ignore-unfixed --skip-files /usr/local/bin/trivy --severity CRITICAL,HIGH --exit-code 1 / \
    && touch /scan-ok

# Final stage - identical to runtime but depends on successful scan
FROM base AS final

COPY --from=scanner /scan-ok /scan-ok
