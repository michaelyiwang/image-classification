# Import required libraries
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.datasets import cifar10
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.regularizers import l2
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Split training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)

# Class names for CIFAR-10
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Convert data type to float32
X_train = X_train.astype("float32")
X_val = X_val.astype("float32")
X_test = X_test.astype("float32")

# Normalize using training set statistics
mean = np.mean(X_train)
std = np.std(X_train)

X_train = (X_train - mean) / (std + 1e-7)
X_val = (X_val - mean) / (std + 1e-7)
X_test = (X_test - mean) / (std + 1e-7)

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)
y_test = to_categorical(y_test, 10)

# Define data augmentation pipeline
data_generator = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.12,
    height_shift_range=0.12,
    horizontal_flip=True,
    zoom_range=0.1,
    brightness_range=[0.9, 1.1],
    shear_range=10,
    channel_shift_range=0.1,
)

# Initialize the CNN model
model = Sequential()
weight_decay = 0.0001

# Block 1
model.add(
    Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(weight_decay),
        input_shape=X_train.shape[1:],
    )
)
model.add(BatchNormalization())
model.add(
    Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(weight_decay),
    )
)
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

# Block 2
model.add(
    Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(weight_decay),
    )
)
model.add(BatchNormalization())
model.add(
    Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(weight_decay),
    )
)
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))

# Block 3
model.add(
    Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(weight_decay),
    )
)
model.add(BatchNormalization())
model.add(
    Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(weight_decay),
    )
)
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.4))

# Block 4
model.add(
    Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(weight_decay),
    )
)
model.add(BatchNormalization())
model.add(
    Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(weight_decay),
    )
)
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))

# Output
model.add(Flatten())
model.add(Dense(10, activation="softmax"))

# Print model summary
model.summary()

# Compile the model
batch_size = 64
epochs = 100
optimizer = Adam(learning_rate=0.0005)

model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)

# Define callbacks
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5)
early_stopping = EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
)

# Train the model
model.fit(
    data_generator.flow(X_train, y_train, batch_size=batch_size),
    epochs=epochs,
    callbacks=[reduce_lr, early_stopping],
    validation_data=(X_val, y_val),
    verbose=1,
)

# Plot loss and accuracy curves
plt.figure(figsize=(20, 6))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(model.history.history["loss"], label="Train Loss", color="#8502d1")
plt.plot(model.history.history["val_loss"], label="Validation Loss", color="darkorange")
plt.title("Loss Curve")
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(model.history.history["accuracy"], label="Train Accuracy", color="#8502d1")
plt.plot(
    model.history.history["val_accuracy"],
    label="Validation Accuracy",
    color="darkorange",
)
plt.title("Accuracy Curve")
plt.legend()

plt.show()

# Evaluate model on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Save model
model.save("/kaggle/working/my_model.keras")
