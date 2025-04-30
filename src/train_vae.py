import os
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, Model

# 1️⃣ Generate synthetic IoT data
normal    = np.random.normal(loc=[30,50,0.1], scale=[5,10,0.05], size=(10000,3))
anomalies = np.random.uniform(low=[0,0,0], high=[100,100,1],   size=(500,3))
X         = np.vstack([normal, anomalies])

# 2️⃣ Fit & save scaler on *normal* only
scaler   = MinMaxScaler().fit(normal)
X_scaled = scaler.transform(X)
os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.pkl')

# 3️⃣ Build a simple autoencoder
inp     = layers.Input(shape=(3,))
h       = layers.Dense(16, activation='relu')(inp)
latent  = layers.Dense(8, activation='relu')(h)
h2      = layers.Dense(16, activation='relu')(latent)
out     = layers.Dense(3, activation='sigmoid')(h2)
ae      = Model(inp, out, name='autoencoder')

# 4️⃣ Compile & train with just MSE
ae.compile(optimizer='adam', loss='mse')
ae.fit(
    X_scaled, X_scaled,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    verbose=2
)

# 5️⃣ Save model
ae.save('models/autoencoder.h5')
print("✅ Trained & saved AE to models/autoencoder.h5")
