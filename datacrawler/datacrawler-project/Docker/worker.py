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
    os.environ["DATAMODULE"] = data[2]

    subprocess.call("datacrawler", shell=True)

    q.complete(item)
  else:
    print("Waiting for work")
print("Queue empty, exiting")
