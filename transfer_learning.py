import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping

# 1. Download Flowers Dataset (Official Google Mirror)
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir_path = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)

IMG_SIZE = (160, 160)
BATCH_SIZE = 32

# 2. Load Dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int' # Use 'int' for SparseCategoricalCrossentropy
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

# 3. Base Model
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,),
                         include_top=False,
                         weights='imagenet')
base_model.trainable = False

# 5. Build Model (Fixed Preprocessing)
model = models.Sequential([
    layers.Input(shape=(160, 160, 3)),
    # This specific Rescaling matches MobileNetV2 expectations
    layers.Rescaling(1./127.5, offset=-1), 
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(5, activation='softmax')
])

# 6. Compile (Ensuring Loss matches Label Mode)
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# 7. Early Stopping
early_stop = EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True)

print("Starting Training...")
model.fit(train_ds, epochs=10, validation_data=val_ds, callbacks=[early_stop])

# --- Fine-Tuning ---
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Starting Fine-Tuning...")
model.fit(train_ds, epochs=5, validation_data=val_ds, callbacks=[early_stop])

# --- Saving ---
model.save('flower_model_final.keras')
print("Model saved as flower_model_final.keras")