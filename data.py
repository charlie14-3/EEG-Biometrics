import pandas as pd
import numpy as np

# 1. Generating a 'Flatline' (Variance near 0)
# This will trigger: "🔴 ERROR: Flatline Detected. No biological liveness."
flatline_data = np.full((1250, 3), 10) # 3 columns, 5 seconds of data
df_flat = pd.DataFrame(flatline_data, columns=['CH1', 'CH2', 'CH3'])
df_flat.to_csv('login_attempts/test_flatline.csv', index=False)

# 2. Generating 'Extreme Noise' (Variance > 5000)
# This will trigger: "🔴 ERROR: Extreme noise detected. Check headset connection."
noise_data = np.random.normal(0, 1000, (1250, 3)) 
df_noise = pd.DataFrame(noise_data, columns=['CH1', 'CH2', 'CH3'])
df_noise.to_csv('login_attempts/test_noise.csv', index=False)

print("Test files 'test_flatline.csv' and 'test_noise.csv' created.")