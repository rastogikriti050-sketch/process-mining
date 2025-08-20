import gzip
import shutil

with gzip.open("Sepsis Cases - Event Log.xes.gz", "rb") as f_in:
    with open("Sepsis_Cases_Log.xes", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
