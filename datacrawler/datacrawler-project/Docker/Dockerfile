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
  xvfb

COPY datacrawler/ /opt/apt/datacrawler/
COPY run_datacrawler.sh /opt/apt/

RUN chmod +x /opt/apt/run_datacrawler.sh
RUN chmod +x /opt/apt/datacrawler/datacrawler

# To analyse given domain, set URL env as follows:
# docker run -e URL=<url> ...
ENTRYPOINT ["/bin/bash", "/opt/apt/run_datacrawler.sh"]
