import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import pathlib

# Paths to your dataset
train_folder = 'C:/Users/Student/.keras/datasets/Alzheimers Dataset/train'
val_folder = 'C:/Users/Student/.keras/datasets/Alzheimers Dataset/validation'

# Image size and batch size
image_height = 160
image_width = 160
batch_size = 32

# Load datasets
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_folder, shuffle=True, batch_size=batch_size, image_size=(image_height, image_width))

val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_folder, shuffle=True, batch_size=batch_size, image_size=(image_height, image_width))

# Get class names
class_names = train_dataset.class_names

# Display some images from the training dataset
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

# Build the CNN model
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(image_height, image_width, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

# Train the model
epochs = 10
history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

# Save the model
model.save('C:/Users/Student/.keras/datasets/Alzheimers Dataset/alzheimers_cnn_model.h5')

# Plot training & validation accuracy values
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
