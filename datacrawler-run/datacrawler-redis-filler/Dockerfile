FROM redis:latest

COPY top100000.csv /opt/apt/
COPY push_urls.sh /opt/apt/

ENTRYPOINT ["/bin/bash", "/opt/apt/push_urls.sh"]



