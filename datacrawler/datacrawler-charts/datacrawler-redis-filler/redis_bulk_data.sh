# !/bin/bash

var=0
cat top100000_plain.csv | while read line
do 
   echo "rpush datacrawler-urls \"$line,SCREENSHOT_MODULE\" " >> top100000_redis.csv
var=$((var + 1))
if [ "$var" -eq "$1" ]
then
	exit 1
fi

done
