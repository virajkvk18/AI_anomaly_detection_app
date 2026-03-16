import requests
import random
import time

API = "http://127.0.0.1:8000/predict"

for i in range(20):

    cpu = random.uniform(10, 100)
    memory = random.uniform(20, 90)
    disk = random.uniform(10, 80)

    params = {
        "cpu_usage": cpu,
        "memory_usage": memory,
        "disk_io": disk
    }

    r = requests.post(API, params=params)

    print("Request:", i + 1)

    try:
        print(r.json())
    except:
        print("Server returned:", r.text)

    time.sleep(1)