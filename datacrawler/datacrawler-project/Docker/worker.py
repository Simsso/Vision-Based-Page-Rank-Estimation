#!/usr/bin/env python
import subprocess
import rediswq
import os

host="redis"

q = rediswq.RedisWQ(name="datacrawler-urls", host="redis")
print("Worker with sessionID: " +  q.sessionID())
print("Initial queue state: empty=" + str(q.empty()))
while not q.empty():
  item = q.lease(lease_secs=10, block=True, timeout=2) 
  if item is not None:
    itemstr = item.decode("utf=8")
    print("Working on " + itemstr)
    data = itemstr.split(",")
    os.environ["URL_RANK"] = data[0]
    os.environ["URL"] = data[1]
    os.environ["DATAMODULE"] = data[3]
    os.environ["OUTPUT_PATH"] = "/opt/apt/output"
    os.environ["ONPAINT_TIMEOUT"] = "25"
    os.environ["ELAPSED_TIME_ONPAINT_TIMEOUT"] = "17500"
    os.environ["CHANGE_THRESHOLD"] = "0.005"
    os.environ["LAST_SCREENSHOTS"] = "20"

    subprocess.call(["/opt/apt/datacrawler/datacrawler", "--no-sandbox", "--headless", "--disable-gpu","--disk-cache-dir=/dev/null", "--disk-cache-size=1"])

    q.complete(item)
  else:
    print("Waiting for work")
print("Queue empty, exiting")
