import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Load and Preprocess
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 2. Build Model Architecture
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1,  activation='sigmoid')
])

# 3. Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 4. Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# 5. Train
print("\nStarting Local Training...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    validation_split=0.2,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# 6. Evaluate on test set
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss     : {loss:.4f}")
print(f"Test Accuracy : {acc:.4f}")

# 7. Save in both formats
model.save('keras_model.h5')       # Legacy HDF5
model.save('keras_model.keras')    # Native Keras format
print("\nModels saved: keras_model.h5 | keras_model.keras")

# 8. Load and verify saved model
loaded = load_model('keras_model.h5')
_, loaded_acc = loaded.evaluate(X_test, y_test, verbose=0)
print(f"Loaded model accuracy : {loaded_acc:.4f}")

# 9. Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['loss'],     label='Train Loss')
ax1.plot(history.history['val_loss'], label='Val Loss',  linestyle='--')
ax1.set_title('Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Binary Cross-entropy')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(history.history['accuracy'],     label='Train Acc')
ax2.plot(history.history['val_accuracy'], label='Val Acc',  linestyle='--')
ax2.set_title('Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(alpha=0.3)

plt.suptitle('Keras Neural Network - Training History', fontweight='bold')
plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
plt.show()
print("Plot saved: training_history.png")