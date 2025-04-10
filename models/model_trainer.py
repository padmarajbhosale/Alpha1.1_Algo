import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from dotenv import load_dotenv

# --- Load Config ---
load_dotenv()
DATA_FILE = os.getenv("TRAINING_FEATURE_FILE", "data/EURUSD_M5_model_ready_features.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.keras")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")
CLASS_WEIGHTS = float(os.getenv("CLASS_1_WEIGHT", 1.0))
IS_BINARY = os.getenv("BINARY_CLASSIFICATION", "true").lower() == "true"

# --- Load Data ---
print(f"📥 Loading data from: {DATA_FILE}")
df = pd.read_csv(DATA_FILE)

X = df.drop(columns=["Next_Return", "Return_Label"], errors='ignore')
X['Regime'] = X.get('Regime', 0).astype(int)
X['OB_Type_Encoded'] = X.get('OB_Type_Encoded', 0).fillna(0)

if IS_BINARY:
    df['Target'] = df['Return_Label'].isin(['Small Gain', 'Big Gain']).astype(int)
    y = df['Target']
    num_classes = 2
else:
    y = df['Return_Label'].astype('category').cat.codes
    num_classes = len(np.unique(y))

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# --- Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Model ---
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
    metrics=['accuracy']
)

# --- Training ---
y_train_cat = to_categorical(y_train, num_classes=num_classes) if num_classes > 2 else y_train
y_test_cat = to_categorical(y_test, num_classes=num_classes) if num_classes > 2 else y_test

print("🚀 Training model...")
model.fit(
    X_train_scaled, y_train_cat,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    class_weight={0: 1.0, 1: CLASS_WEIGHTS} if IS_BINARY else None,
    verbose=1
)

# --- Evaluation ---
print("\n📊 Classification Report:")
y_pred = model.predict(X_test_scaled)
if num_classes > 2:
    y_pred_classes = np.argmax(y_pred, axis=1)
else:
    y_pred_classes = (y_pred > 0.5).astype(int)

print(classification_report(y_test, y_pred_classes))

# --- Save ---
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print(f"\n✅ Model saved: {MODEL_PATH}")
print(f"✅ Scaler saved: {SCALER_PATH}")
