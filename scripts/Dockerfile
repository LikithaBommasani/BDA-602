#Reference: https://docker-curriculum.com/#dockerfile
FROM python:3.9.6

MAINTAINER likithabommasani

ENV APP_HOME /app
WORKDIR $APP_HOME
ENV PYTHONPATH /

# Get necessary system packages
RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
     python3 \
     python3-pip \
     python3-dev \
  && rm -rf /var/lib/apt/lists/*

# Get necessary python libraries
COPY requirements.txt .
RUN pip3 install --compile --no-cache-dir -r requirements.txt

## Copy over code
#COPY ./baseball_script.sql/app/baseball_script.sql
#COPY
#
#
#
## Run app
#CMD
