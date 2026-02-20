import tensorflow as tf
from model import build_cnn_model

dataset_path = "data/raw/plantvillage_dataset/color"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

# -----------------------------
# LOAD DATASET
# -----------------------------

train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names
num_classes = len(class_names)

# -----------------------------
# DATA AUGMENTATION (Dataset 2)
# -----------------------------

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
])

train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y))

# -----------------------------
# NORMALISATION
# -----------------------------

normalization_layer = tf.keras.layers.Rescaling(1./255)

train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

# -----------------------------
# PERFORMANCE OPTIMISATION
# -----------------------------

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# -----------------------------
# BUILD MODEL
# -----------------------------

model = build_cnn_model(num_classes)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# EARLY STOPPING
# -----------------------------

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# -----------------------------
# TRAIN
# -----------------------------

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=30,
    callbacks=[early_stop]
)

# -----------------------------
# SAVE MODEL
# -----------------------------

model.save("models/plant_cnn.h5")
