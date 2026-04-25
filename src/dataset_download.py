import requests
import pandas as pd
import os
import time
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
# 1. Output Folder
OUTPUT_FOLDER = "tamil_nadu_weather_data"

# 2. Date Range (Adjust as needed for your training history)
START_DATE = "20150101"  # Format: YYYYMMDD
END_DATE = "20231231"    # Format: YYYYMMDD

# 3. Parameters required for your GCN-Transformer Model
# T2M: Temperature at 2 Meters (C)
# RH2M: Relative Humidity at 2 Meters (%)
# WS2M: Wind Speed at 2 Meters (m/s)
# WD2M: Wind Direction at 2 Meters (Degrees) - Critical for Spatial Graph Edges
# PRECTOTCORR: Precipitation Corrected (mm/day) - Target Variable
# CLOUD_AMT: Cloud Amount (%) - As per your senior's requirement
PARAMETERS = "T2M,RH2M,WS2M,WD2M,PRECTOTCORR,CLOUD_AMT"

# 4. Target Locations: All Major Districts of Tamil Nadu (Lat/Lon)
# These coordinates are central points for the districts.
LOCATIONS = {
    "Chennai":        {"lat": 13.0827, "lon": 80.2707},
    "Coimbatore":     {"lat": 11.0168, "lon": 76.9558},
    "Madurai":        {"lat": 9.9252,  "lon": 78.1198},
    "Tiruchirappalli": {"lat": 10.7905, "lon": 78.7047},
    "Salem":          {"lat": 11.6643, "lon": 78.1460},
    "Tirunelveli":    {"lat": 8.7139,  "lon": 77.7567},
    "Erode":          {"lat": 11.3410, "lon": 77.7172},
    "Vellore":        {"lat": 12.9165, "lon": 79.1325},
    "Thoothukudi":    {"lat": 8.7642,  "lon": 78.1348},
    "Dindigul":       {"lat": 10.3673, "lon": 77.9803},
    "Thanjavur":      {"lat": 10.7870, "lon": 79.1378},
    "Tiruppur":       {"lat": 11.1085, "lon": 77.3411},
    "Virudhunagar":   {"lat": 9.5680,  "lon": 77.9624},
    "Karur":          {"lat": 10.9601, "lon": 78.0766},
    "Nilgiris":       {"lat": 11.4916, "lon": 76.7337}, # Ooty
    "Kancheepuram":   {"lat": 12.8185, "lon": 79.6947},
    "Kanyakumari":    {"lat": 8.0883,  "lon": 77.5385},
    "Cuddalore":      {"lat": 11.7480, "lon": 79.7714},
    "Dharmapuri":     {"lat": 12.1211, "lon": 78.1582},
    "Villupuram":     {"lat": 11.9401, "lon": 79.4861}
}

# ==========================================
# DATA FETCHING FUNCTION
# ==========================================
def fetch_nasa_power_data(city_name, lat, lon):
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    params = {
        "parameters": PARAMETERS,
        "community": "AG", # Agroclimatology (best for rain/weather)
        "longitude": lon,
        "latitude": lat,
        "start": START_DATE,
        "end": END_DATE,
        "format": "JSON"
    }

    print(f"Fetching data for {city_name}...")
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status() # Check for errors
        data = response.json()
        
        # Parse the nested JSON response
        properties = data['properties']['parameter']
        
        # Create a DataFrame
        df_list = []
        # The API returns data as dictionary {date: value}. We need to align them.
        # We grab the dates from the first parameter (e.g., T2M)
        dates = sorted(properties['T2M'].keys())
        
        for date in dates:
            row = {'Date': date}
            for param in PARAMETERS.split(','):
                # Handle missing values (-999 is NASA's null value)
                val = properties[param].get(date, None)
                row[param] = val if val != -999 else None 
            df_list.append(row)
            
        df = pd.DataFrame(df_list)
        
        # Convert Date string YYYYMMDD to datetime
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
        
        # Save to CSV
        file_path = os.path.join(OUTPUT_FOLDER, f"{city_name}.csv")
        df.to_csv(file_path, index=False)
        print(f"✔ Saved: {file_path}")
        
    except Exception as e:
        print(f"✘ Error fetching {city_name}: {e}")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        
    print(f"Starting download for {len(LOCATIONS)} districts...")
    print(f"Parameters: {PARAMETERS}")
    print("-" * 30)

    for city, coords in LOCATIONS.items():
        fetch_nasa_power_data(city, coords['lat'], coords['lon'])
        # Small delay to be polite to the NASA server
        time.sleep(1.5)

    print("-" * 30)
    print("Download Complete! Check the folder:", OUTPUT_FOLDER)