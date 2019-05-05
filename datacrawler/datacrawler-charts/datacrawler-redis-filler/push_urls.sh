# !/bin/bash

echo "Pushing URLs in to job queue!"
cat /opt/apt/top100000_redis.csv | redis-cli -h redis --pipe 
echo "Pushing URLs has finished!"
