import requests
import pandas as pd
from datetime import datetime

# Bay of Bengal ocean point
LAT = 12.0
LON = 85.0

url = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
    f"?parameters=TS"          # Earth Skin Temperature (acts like SST over ocean)
    f"&community=RE"           # Renewable Energy community (supports TS)
    f"&longitude={LON}"
    f"&latitude={LAT}"
    f"&start=20000101"
    f"&end=20231231"
    f"&format=JSON"
)

print("Requesting TS (SST-like) data from NASA POWER...")

resp = requests.get(url)
data = resp.json()

# Safety check – show error if POWER returns an error structure
if "properties" not in data:
    print("ERROR from NASA POWER:")
    print(data)
    raise SystemExit("API call failed – check printed message above.")

daily_data = data["properties"]["parameter"]["TS"]

rows = []
for date_str, val in daily_data.items():
    date = datetime.strptime(date_str, "%Y%m%d")
    kelvin = float(val)
    celsius = kelvin - 273.15
    rows.append([date, celsius])

df = pd.DataFrame(rows, columns=["Date", "SST"])
df.to_csv("sst_daily.csv", index=False)

print("✔ SST proxy saved as sst_daily.csv")
