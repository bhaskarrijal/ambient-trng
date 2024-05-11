import numpy as np
from sklearn.decomposition import PCA

# Sensor Data
sensor_data = np.array([
    [0.5, 50, 2700, 0.5],  # Time 0
    [0.6, 55, 2700, 0.6],  # Time 1
    [0.7, 60, 2700, 0.7],  # Time 2
    [0.65, 58, 2700, 0.65],  # Time 3
    [0.55, 52, 2700, 0.55],  # Time 4
    [0.5, 48, 2700, 0.5],  # Time 5
    [0.45, 45, 2700, 0.45],  # Time 6
    [0.4, 42, 2700, 0.4],  # Time 7
    [0.35, 38, 2700, 0.35],  # Time 8
    [0.3, 35, 2700, 0.3],  # Time 9
    [0.25, 32, 2700, 0.25],  # Time 10
    [0.2, 30, 2700, 0.2],  # Time 11
    [0.15, 28, 2700, 0.15],  # Time 12
    [0.1, 26, 2700, 0.1],  # Time 13
    [0.05, 24, 2700, 0.05],  # Time 14
    [0, 22, 2700, 0],  # Time 15
])

# Method 1: Noise-based Randomization
noise = sensor_data - np.mean(sensor_data)
# Ensure the seed is within the valid range
seed = int(noise[0, 0]) % (2**32 - 1)
rng = np.random.RandomState(seed)

# Method 2: Spectral Response-based Randomization
pca = PCA()  # Create an instance of PCA
features = pca.fit_transform(sensor_data)
# Ensure the seed is within the valid range
seed = int(features[0, 0]) % (2**32 - 1)
rng = np.random.RandomState(seed)

# Method 3: Intensity-based Randomization
intensity = sensor_data[:, 1]  # Assuming intensity is the second column
# Ensure the seed is within the valid range
seed = int(np.round(np.mean(intensity))) % (2**32 - 1)
rng = np.random.RandomState(seed)

# Method 4: Hybrid Approach
combined_data = np.concatenate((sensor_data, np.random.rand(len(sensor_data), sensor_data.shape[1])), axis=1)
# Ensure the seed is within the valid range
seed = int(np.round(np.mean(combined_data))) % (2**32 - 1)
rng = np.random.RandomState(seed)

print("Random Value:", rng.rand())
