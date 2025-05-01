import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam

# --- 1) Synthetic data generation ---
np.random.seed(42)
# Normal / anomaly split
normal = np.random.normal(loc=[30,50,0.1], scale=[5,10,0.05], size=(10000,3))
anomalies = np.random.uniform(low=[0,0,0], high=[100,100,1], size=(500,3))
# Train / val / test
train_normal = normal[:8000]
val_normal   = normal[8000:9000]
test_normal  = normal[9000:]
X_test       = np.vstack([test_normal, anomalies])
y_test       = np.array([0]*len(test_normal) + [1]*len(anomalies))

# --- 2) Scaling ---
scaler = MinMaxScaler().fit(train_normal)
X_train = scaler.transform(train_normal)
X_val   = scaler.transform(val_normal)
X_test  = scaler.transform(X_test)

os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.pkl')

# --- 3) Build autoencoder ---
inp    = layers.Input(shape=(3,))
h      = layers.Dense(16, activation='relu')(inp)
latent = layers.Dense(8, activation='relu')(h)
h2     = layers.Dense(16, activation='relu')(latent)
out    = layers.Dense(3, activation='sigmoid')(h2)
ae     = Model(inp, out, name='ae')
ae.compile(optimizer=Adam(1e-3), loss='mse')

# --- 4) Train ---
history = ae.fit(
    X_train, X_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_val, X_val),
    verbose=2
)
ae.save('models/autoencoder.h5')

# --- 5) Plot & save loss curves ---
plt.figure()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend()
plt.tight_layout()
plt.savefig('loss_curves.png')

# --- 6) Histograms of normal features ---
fig, axs = plt.subplots(1,3,figsize=(12,4))
for i, name in enumerate(['Temp (°C)','Hum (%)','Vib (g)']):
    axs[i].hist(test_normal[:,i], bins=30, alpha=0.7)
    axs[i].set_title(name)
plt.tight_layout()
plt.savefig('normal_histograms.png')

# --- 7) Reconstruction & errors ---
recon = ae.predict(X_test, verbose=0)
errs  = np.mean((X_test - recon)**2, axis=1)
mu, sigma = errs[:len(test_normal)].mean(), errs[:len(test_normal)].std()
thresh = mu + 3*sigma

# --- 8) Metrics ---
y_pred = (errs > thresh).astype(int)
accuracy = (y_pred == y_test).mean()
fpr = (y_pred[:len(test_normal)]==1).sum() / len(test_normal)

# --- 9) Noise robustness ---
noise = np.random.normal(scale=0.02, size=test_normal.shape)
noisy = scaler.transform(test_normal + noise)
recon_noisy = ae.predict(noisy, verbose=0)
errs_noisy  = np.mean((noisy - recon_noisy)**2, axis=1)
var_inc = (errs_noisy.var() - errs[:len(test_normal)].var()) / errs[:len(test_normal)].var() * 100

# --- 10) Threshold sensitivity sweep ---
mul = np.arange(2.5,3.51,0.25)
sens = []
for m in mul:
    t = mu + m*sigma
    p = (errs > t).astype(int)
    sens.append({
      'mult': m,
      'TPR': (p[len(test_normal):]==1).mean(),
      'FPR': (p[:len(test_normal)]==1).mean()
    })
sens_df = pd.DataFrame(sens)

# --- 11) Inference timing & sizes ---
start = time.perf_counter()
_ = ae.predict(X_test, verbose=0)
elapsed = (time.perf_counter() - start) / len(X_test) * 1000
model_mb  = os.path.getsize('models/autoencoder.h5')/1024**2
scaler_mb = os.path.getsize('models/scaler.pkl')/1024**2

# --- 12) Error distribution plot ---
plt.figure()
plt.hist(errs[:len(test_normal)], bins=50, alpha=0.6, label='normal')
plt.hist(errs[len(test_normal):], bins=50, alpha=0.6, label='anomaly')
plt.axvline(thresh, color='k', linestyle='--', label='threshold')
plt.xlabel('MSE error'); plt.ylabel('count'); plt.legend()
plt.tight_layout()
plt.savefig('error_distribution.png')

# --- 13) Print summary ---
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"False-positive rate: {fpr*100:.2f}%")
print(f"Noise variance ↑: {var_inc:.2f}%")
print(f"Avg inference: {elapsed:.2f} ms/sample")
print(f"Model size: {model_mb:.2f} MB; Scaler: {scaler_mb:.2f} MB")
print("\nThreshold sensitivity:")
print(sens_df.to_string(index=False))
