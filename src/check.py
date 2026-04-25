import numpy as np

X = np.load("processed_data/X.npy")
rainfall = X[:,:,5]  # rainfall column

rain_days = (rainfall > 0.2).sum()
dry_days = (rainfall <= 0.2).sum()

print("Rain days:", rain_days)
print("Dry days:", dry_days)
print("Rain %:", rain_days / (rain_days + dry_days) * 100)