import psutil
import pandas as pd

data=[]

for i in range(100):

    cpu=psutil.cpu_percent()
    memory=psutil.virtual_memory().percent
    disk=psutil.disk_usage('/').percent

    data.append([cpu,memory,disk])

df=pd.DataFrame(data,columns=["cpu_usage","memory_usage","disk_io"])

df.to_csv("data/system_metrics.csv",index=False)