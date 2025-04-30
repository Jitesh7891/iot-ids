import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# 1️⃣ Load your CSV data
df = pd.read_csv('data/random_iot_data_1000.csv')
X  = df[['temp (°C)', 'hum (%)', 'vib (g)']].values

# 2️⃣ Load the scaler from training
scaler   = joblib.load('models/scaler.pkl')
X_scaled = scaler.transform(X)

# 3️⃣ Load the trained autoencoder
ae = load_model('models/autoencoder.h5', compile=False)

# 4️⃣ Reconstruct & compute per-sample MSE
recon  = ae.predict(X_scaled, verbose=0)
errors = np.mean((X_scaled - recon)**2, axis=1)

# 5️⃣ Choose threshold: mean + 3·std
mu, sigma = errors.mean(), errors.std()
threshold = mu + 3 * sigma

# 6️⃣ Print anomaly flags and count
anomaly_count = 0
normal_count = 0

for i, err in enumerate(errors):
    flag = 'ANOMALY' if err > threshold else 'normal'
    print(f"Sample {i:3d}: {flag:8s} (MSE={err:.5f})")
    if flag == 'ANOMALY':
        anomaly_count += 1
    else:
        normal_count += 1

print(f"\n✅ Total: {len(errors)} samples")
print(f"🔴 Anomalies: {anomaly_count}")
print(f"🟢 Normal:    {normal_count}")
