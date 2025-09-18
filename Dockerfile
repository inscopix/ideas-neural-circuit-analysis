FROM python:3.13.5

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONFAULTHANDLER=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && \
    apt-get install -y libgl1

COPY requirements.txt ./
RUN python -m pip install -r requirements.txt
