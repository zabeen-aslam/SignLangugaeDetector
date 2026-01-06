# -------------------------
# 1. MOUNT GOOGLE DRIVE
# -------------------------
from google.colab import drive
drive.mount('/content/drive')

# -------------------------
# 2. IMPORT LIBRARIES
# -------------------------
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import classification_report

# -------------------------
# 3. REPRODUCIBILITY
# -------------------------
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# -------------------------
# 4. DEFINE PATHS (already sampled dataset)
# -------------------------
small_train = "/content/drive/MyDrive/SmallDataset/Train"
small_val   = "/content/drive/MyDrive/SmallDataset/Val"
small_test  = "/content/drive/MyDrive/SmallDataset/Test"

# -------------------------
# 5. DATA LOADING (GRAYSCALE)
# -------------------------
img_size = (32,32)
batch_size = 32
epochs = 20

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    small_train,
    target_size=img_size,
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    small_val,
    target_size=img_size,
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True
)

test_gen = test_datagen.flow_from_directory(
    small_test,
    target_size=img_size,
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

num_classes = len(train_gen.class_indices)

# -------------------------
# 6. CNN MODEL
# -------------------------
inputs = layers.Input(shape=(32,32,1))

x = layers.Conv2D(32, (3,3), activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(64, (3,3), activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(128, (3,3), activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs)

# -------------------------
# 7. COMPILE MODEL
# -------------------------
model.compile(
    optimizer=optimizers.Adam(0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------------
# 8. TRAIN MODEL
# -------------------------
print("\n========== TRAINING STARTED ==========\n")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    verbose=1
)

# -------------------------
# 9. TEST MODEL
# -------------------------
print("\n========== TESTING MODEL ==========\n")
test_loss, test_acc = model.evaluate(test_gen, verbose=1)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")
print(f"Final Test Loss:     {test_loss:.4f}")

# -------------------------
# 10. CLASSIFICATION REPORT
# -------------------------
preds = model.predict(test_gen)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes
labels = list(test_gen.class_indices.keys())

print("\n========== CLASSIFICATION REPORT ==========\n")
print(classification_report(y_true, y_pred, target_names=labels))

# -------------------------
# 11. SAVE MODEL
# -------------------------
model.save("/content/drive/MyDrive/Fast_Gray_SignLanguageCNN.h5")
print("\nModel saved as Fast_Gray_SignLanguageCNN.h5")
