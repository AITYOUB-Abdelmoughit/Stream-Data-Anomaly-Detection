import numpy as np
import time

def simulate_data_stream(length=1000, noise=0.2):
    time_steps = np.arange(length)
    seasonal_pattern = np.sin(time_steps / 50)  # Simulate seasonal variations
    data_stream = seasonal_pattern + np.random.normal(0, noise, length)

    for data_point in data_stream:
        time.sleep(0.1)  # Simulate real-time delay
        yield data_point
