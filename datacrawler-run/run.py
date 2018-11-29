import csv
import docker
import time
from multiprocessing import Pool, Process

client = docker.from_env()


def compute(data):
    environment = dict()
    environment["URL_RANK"] = data[0]
    environment["URL"] = data[1]
    environment["DATAMODULE"] = "SCREENSHOT_MODULE"
    environment["OUTPUT_PATH"] = "/opt/apt/output"
    environment["ONPAINT_TIMEOUT"] = 25
    environment["ELAPSED_TIME_ONPAINT_TIMEOUT"] = 17500
    environment["CHANGE_THRESHOLD"] = 0.005
    environment["LAST_SCREENSHOTS"] = 20
    container = client.containers.run("datacrawler", detach=True, environment=environment, remove=True, volumes={
            '/home/doktorgibson/Desktop/Pagerank-Estimation-Deep-Graph-Networks/datacrawler-run/output/': {'bind': '/opt/apt/output', 'mode': 'rw'}})

    container.wait()

    print(data[0] + " finished!")

with open('top100.csv', 'rt',  encoding="utf8") as csvfile:
    reader = csv.reader(csvfile)
    data = []

    for row in reader:
        data.append((row[0], row[1]))

    pool = Pool(3)
    pool.map(compute, data)

   # docker run -e URL=google.de -e URL_RANK=1 -e OUTPUT_PATH=/opt/apt/output -e DATAMODULE=SCREENSHOT_MODULE -e ONPAINT_TIMEOUT=20 -e ELAPSED_TIME_ONPAINT_TIMEOUT=15 -e CHANGE_THRESHOLD=0.0005 -e LAST_SCREENSHOTS=20 -v /home/doktorgibson/Desktop/run/test-output/:/opt/apt/output datacrawler
