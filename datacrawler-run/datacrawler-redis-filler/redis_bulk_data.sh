# !/bin/bash

var=0
cat oldtop100000.csv | while read line
do 
   echo "rpush datacrawler-urls \"$line,SCREENSHOT_MODULE\" " >> 100000.csv
var=$((var + 1))
if [ "$var" -eq "$1" ]
then
	exit 1
fi

done
