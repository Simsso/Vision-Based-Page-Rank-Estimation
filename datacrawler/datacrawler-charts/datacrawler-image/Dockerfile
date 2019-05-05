FROM ubuntu:16.04

RUN apt-get update && apt-get install -y \
  build-essential \
  libgtk2.0-dev \
  libnss3-dev \
  libgconf-2-4 \
  libxss-dev \
  libasound2-dev \
  libxtst-dev \
  libgl1-mesa-dev \
  libatk-bridge2.0-0 \
  libtbb-dev \
  xvfb \
  tzdata && \
    rm /etc/localtime && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get clean

RUN apt-get install -y  python-pip
RUN pip install redis 

COPY datacrawler-bin/ /opt/apt/datacrawler/
RUN chmod 777 /opt/apt/datacrawler/datacrawler
COPY rediswq.py /opt/apt/
COPY worker.py  /opt/apt/
COPY run_datacrawler.sh /opt/apt/

RUN mkdir /opt/apt/output
RUN cat /opt/apt/datacrawler/datacrawler.config.json
RUN chmod 777 /opt/apt/datacrawler/datacrawler.config.json
RUN chmod 777 /opt/apt/run_datacrawler.sh
RUN chmod +x /opt/apt/run_datacrawler.sh
RUN chmod +x /opt/apt/datacrawler/datacrawler

ENTRYPOINT ["/bin/bash", "/opt/apt/run_datacrawler.sh"]

