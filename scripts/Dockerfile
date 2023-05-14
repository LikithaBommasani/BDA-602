#Reference: https://docker-curriculum.com/#dockerfile
FROM python:3.10

MAINTAINER likithabommasani

ENV APP_HOME /app
WORKDIR /app
ENV PYTHONPATH /

# Refernce : https://www.reddit.com/r/mariadb/comments/y28yf5/unable_to_install_mariadb_in_python_docker/

RUN apt-get install wget && \
  wget https://downloads.mariadb.com/MariaDB/mariadb_repo_setup && \
  echo "733cf126b03f73050e242102592658913d10829a5bf056ab77e7f864b3f8de1f  mariadb_repo_setup" sha256sum -c - && \
  chmod +x mariadb_repo_setup && \
  ./mariadb_repo_setup --mariadb-server-version="mariadb-10.6"


# Get necessary system packages
RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
     python3 \
     python3-pip \
     python3-dev \
     libmariadb3 libmariadb-dev mariadb-client\
  && rm -rf /var/lib/apt/lists/*


# Get necessary python libraries
COPY requirements.txt .

RUN pip3 install --compile --no-cache-dir -r requirements.txt

## Copy over code
COPY baseball.sql $APP_HOME
COPY baseball_features.sql $APP_HOME
COPY my_script.sh $APP_HOME
COPY Hw_06.py $APP_HOME
COPY msd_hw_06.py $APP_HOME
COPY correlation_hw_06.py $APP_HOME


#RUN chmod +x ./my_script.sh
#CMD ["/bin/bash", "/my_script.sh"]
