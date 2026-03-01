import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

# 1. Load Dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 2. Reshape and Normalize (samples, height, width, channels)
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 3 & 4. Build CNN Architecture
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)), # Explicit Input layer prevents 'never been called' errors
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 5. Compile Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 6. Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(x_train)

# 7. Train Model
# Note: 938 steps per epoch is normal (60,000 / 64 = 938)
history = model.fit(datagen.flow(x_train, y_train, batch_size=64),
                    epochs=5, validation_data=(x_test, y_test))

# 8. Visualize First Layer Filters
filters, biases = model.layers[0].get_weights()
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

plt.figure(figsize=(8, 8))
for i in range(16):
    f = filters[:, :, 0, i]
    plt.subplot(4, 4, i+1)
    plt.imshow(f, cmap='gray')
    plt.axis('off')
plt.suptitle("First Layer Filters")
plt.show()

# 9. Visualize Feature Maps (Updated to fix AttributeError)
def visualize_feature_maps(model, image):
    # Select the output of the first 4 layers
    layer_outputs = [layer.output for layer in model.layers[:4]]
    
    # Create a model that returns these outputs given the model input
    # Using model.inputs (plural) handles Keras 3.0+ requirements better
    activation_model = models.Model(inputs=model.inputs, outputs=layer_outputs)
    
    # Prepare image and predict
    img_tensor = image.reshape(1, 28, 28, 1)
    activations = activation_model.predict(img_tensor)
    
    # Display the feature maps of the first convolution layer
    first_layer_activation = activations[0]
    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(first_layer_activation[0, :, :, i], cmap='viridis')
        plt.axis('off')
    plt.suptitle("Feature Maps (First Conv Layer)")
    plt.show()

visualize_feature_maps(model, x_test[0])

# 10. Confusion Matrix
y_pred = np.argmax(model.predict(x_test), axis=1)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 11 & 12. Convert to TFLite and Save
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('fashion_mnist_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model saved as fashion_mnist_model.tflite")