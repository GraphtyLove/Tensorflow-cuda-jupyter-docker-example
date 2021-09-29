import os

dataset_folder = "dataset"

# Training data folder
train_dir = os.path.join(dataset_folder, 'train')
# Test data folder
validation_dir = os.path.join(dataset_folder, 'validation')

# Directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')

# Directory with our training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')

# Directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')

# Directory with our validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop

# Model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compiling model
model.compile(
    # Choose the loss function
    loss='binary_crossentropy',
    # Choose your optimizer
    optimizer=RMSprop(learning_rate=1e-4),
    # Choose the metric the model will use to evaluate his learning
    metrics=['accuracy']
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    # This is the source directory for training images    
    train_dir,
    # All images will be resized to 150x150
    target_size=(150, 150),  
    # Define how big are gonna be your batch.
    batch_size=20,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary'
)


# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

# Check if GPU is detected by tensorflow
print("GPU AVAILABLE: ", tf.config.list_physical_devices('GPU'))

import time
print("--------- TRAINING START ---------")

history = model.fit(
    train_generator,
    # 2000 images = batch_size * steps
    steps_per_epoch=100,
    epochs=50,
    validation_data=validation_generator,
    # 1000 images = batch_size * steps
    validation_steps=50,  
    verbose=1
)

print("training DONE.")
print(f"Took {(time.time() - start_time)}s ")