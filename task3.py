import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

# Define image size and path
image_size = (160, 160)
train_folder = 'C:/Users/Student/.keras/datasets/Alzheimers Dataset/train'
val_folder = 'C:/Users/Student/.keras/datasets/Alzheimers Dataset/validation'

# Load datasets
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_folder, shuffle=True, batch_size=32, image_size=image_size)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_folder, shuffle=True, batch_size=32, image_size=image_size)

# Display a sample of images
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(train_dataset.class_names[labels[i]])
        plt.axis("off")
plt.show()

# Load pre-trained MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(input_shape=image_size + (3,), include_top=False, weights='imagenet')

# Add custom layers on top of MobileNetV2
inputs = tf.keras.Input(shape=(160, 160, 3))
x = tf.keras.layers.RandomFlip('horizontal')(inputs)
x = tf.keras.layers.RandomRotation(0.2)(x)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(len(train_dataset.class_names), activation='softmax')(x)

# Define the new model
model = tf.keras.Model(inputs, outputs)

# Set up fine-tuning
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=10)

# Plot training & validation accuracy
plt.figure(figsize=(8, 8))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Fine-tuned Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Plot training & validation loss
plt.figure(figsize=(8, 8))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Fine-tuned Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
