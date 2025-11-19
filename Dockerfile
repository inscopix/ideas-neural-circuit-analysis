# Create base image to run analysis
FROM public.ecr.aws/docker/library/python:3.13.9 AS base

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

# Copy python project settings
COPY pyproject.toml ./

# Create a venv to install python dependencies
# This can be done globally, but using venv is best practice
RUN apt-get -y update \
    && apt-get install -y libgl1 \
    && ${PYTHON} -m venv ${VENV} \
    && ${PYTHON_VENV} -m pip install --upgrade pip \
    && ${PYTHON_VENV} -m pip install .

ENV IDEAS_PYTHON_WHL_FILE="ideas_python-0.1.dev39+gd34af9b54.d20251119-py3-none-any.whl"
COPY resources/${IDEAS_PYTHON_WHL_FILE} ./
RUN ${PYTHON_VENV} -m pip install "${IDEAS_PYTHON_WHL_FILE}[analysis]"

# Add venv bin to path
ENV PATH="/ideas/${VENV}/bin:${PATH}"

USER ideas
CMD ["/bin/bash"]

# Create image for testing which copies tool code and test data to
# docker image in order to facilitate unit testing in an isolated environment.
# This can also be acheived with volume mounts, but that can clutter up
# your local folder with files generated during testing.
FROM base AS test

COPY --chown=ideas ./ /ideas