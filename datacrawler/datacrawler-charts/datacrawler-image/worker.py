#!/usr/bin/env python
# From https://kubernetes.io/docs/tasks/job/fine-parallel-processing-work-queue/
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
    os.environ["RANK"] = data[0]
    os.environ["URL"] = data[1]

    return_code = subprocess.call(["/opt/apt/datacrawler/datacrawler", "--no-sandbox", "--headless", "--disable-gpu","--disk-cache-dir=/dev/null", "--disk-cache-size=1"])
    
    if return_code == 0:
      q.complete(item)
  else:
    print("Waiting for work")
print("Queue empty, exiting")
