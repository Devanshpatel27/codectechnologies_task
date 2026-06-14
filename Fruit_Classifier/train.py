import tensorflow as tf
from tensorflow.keras import layers, models

# Load Dataset
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(100, 100),
    batch_size=32
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(100, 100),
    batch_size=32
)

# Print Class Names
class_names = train_data.class_names
print("Classes:", class_names)

# Build CNN Model
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(100,100,3)),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),

    layers.Dense(len(class_names), activation='softmax')
])

# Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train Model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15
)

# Save Model
model.save("fruit_classifier.h5")

print("Model Saved Successfully!")