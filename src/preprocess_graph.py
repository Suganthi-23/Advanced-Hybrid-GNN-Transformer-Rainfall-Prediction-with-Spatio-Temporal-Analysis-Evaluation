import pandas as pd
import numpy as np
import os
import glob
from math import radians, cos, sin, asin, sqrt

# ============================
# CONFIGURATION
# ============================
INPUT_FOLDER = "tamil_nadu_weather_data"
OUTPUT_FOLDER = "processed_data"
DISTANCE_THRESHOLD_KM = 150  # Connect stations if they are within 150km

# NEW: Add SST as a feature
FEATURE_COLS = ["T2M", "RH2M", "WS2M", "WD2M", "CLOUD_AMT", "PRECTOTCORR", "SST"]

# ============================
# HELPER: HAVERSINE DISTANCE
# ============================
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

# ============================
# MAIN PROCESSING
# ============================
if __name__ == "__main__":

    # Create output folder
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # Load SST (proxy from NASA POWER)
    print("Loading SST data...")
    sst_df = pd.read_csv("sst_daily.csv")
    sst_df["Date"] = pd.to_datetime(sst_df["Date"])

    all_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
    print(f"Found {len(all_files)} district files.")

    # Reference Dates (common time index)
    ref_df = pd.read_csv(all_files[0])
    ref_df["Date"] = pd.to_datetime(ref_df["Date"])
    reference_dates = ref_df["Date"].sort_values().unique()

    node_features = []
    station_names = []
    station_coords = []

    print("Processing CSVs and aligning dates...")

    # District coordinates
    LOCATIONS = {
        "Chennai": {"lat": 13.0827, "lon": 80.2707},
        "Coimbatore": {"lat": 11.0168, "lon": 76.9558},
        "Madurai": {"lat": 9.9252, "lon": 78.1198},
        "Tiruchirappalli": {"lat": 10.7905, "lon": 78.7047},
        "Salem": {"lat": 11.6643, "lon": 78.1460},
        "Tirunelveli": {"lat": 8.7139, "lon": 77.7567},
        "Erode": {"lat": 11.3410, "lon": 77.7172},
        "Vellore": {"lat": 12.9165, "lon": 79.1325},
        "Thoothukudi": {"lat": 8.7642, "lon": 78.1348},
        "Dindigul": {"lat": 10.3673, "lon": 77.9803},
        "Thanjavur": {"lat": 10.7870, "lon": 79.1378},
        "Tiruppur": {"lat": 11.1085, "lon": 77.3411},
        "Virudhunagar": {"lat": 9.5680, "lon": 77.9624},
        "Karur": {"lat": 10.9601, "lon": 78.0766},
        "Nilgiris": {"lat": 11.4916, "lon": 76.7337},
        "Kancheepuram": {"lat": 12.8185, "lon": 79.6947},
        "Kanyakumari": {"lat": 8.0883, "lon": 77.5385},
        "Cuddalore": {"lat": 11.7480, "lon": 79.7714},
        "Dharmapuri": {"lat": 12.1211, "lon": 78.1582},
        "Villupuram": {"lat": 11.9401, "lon": 79.4861}
    }

    valid_stations = []

    for filename in all_files:
        station_name = os.path.basename(filename).replace(".csv", "")

        if station_name not in LOCATIONS:
            print(f"Skipping {station_name} (No coordinates found)")
            continue

        df = pd.read_csv(filename)
        df["Date"] = pd.to_datetime(df["Date"])

        # Merge SST
        df = df.merge(sst_df, on="Date", how="left")
        df["SST"] = df["SST"].interpolate()

        # Align dates with reference
        df = df.set_index("Date").reindex(reference_dates)
        df = df.interpolate(method="linear", limit_direction="both")

        # Extract feature values
        station_data = df[FEATURE_COLS].values

        node_features.append(station_data)
        station_names.append(station_name)
        station_coords.append(LOCATIONS[station_name])
        valid_stations.append(station_name)

    # Convert to array: (Stations, Days, Features)
    X = np.stack(node_features, axis=0)

    print(f"Feature Matrix Shape: {X.shape} (Stations, Days, Features)")

    # ============================
    # BUILD ADJACENCY MATRIX
    # ============================
    num_stations = len(valid_stations)
    adj_matrix = np.zeros((num_stations, num_stations))

    print("Computing spatial graph edges...")
    edge_count = 0

    for i in range(num_stations):
        for j in range(num_stations):
            if i == j:
                adj_matrix[i][j] = 1
                continue

            lat1, lon1 = station_coords[i]["lat"], station_coords[i]["lon"]
            lat2, lon2 = station_coords[j]["lat"], station_coords[j]["lon"]

            dist = haversine(lon1, lat1, lon2, lat2)

            if dist <= DISTANCE_THRESHOLD_KM:
                adj_matrix[i][j] = 1
                edge_count += 1

    print(f"Graph Construction Complete. Total Edges: {edge_count}")

    # ============================
    # SAVE FINAL DATA
    # ============================
    np.save(os.path.join(OUTPUT_FOLDER, "X.npy"), X)
    np.save(os.path.join(OUTPUT_FOLDER, "A.npy"), adj_matrix)
    np.save(os.path.join(OUTPUT_FOLDER, "station_names.npy"), station_names)

    print("-" * 40)
    print(f"✔ Data ready in '{OUTPUT_FOLDER}/'")
    print("Saved: X.npy, A.npy, station_names.npy")
