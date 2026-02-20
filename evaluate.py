import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

dataset_path = "data/raw/plantvillage_dataset/color"

# Charger dataset validation
val_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(128,128),
    batch_size=32
)

# IMPORTANT : récupérer les noms des classes ici
class_names = val_dataset.class_names

# Normalisation
normalization_layer = tf.keras.layers.Rescaling(1./255)
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

# Charger modèle
model = tf.keras.models.load_model("models/desert_plant_cnn.h5")

y_true = []
y_pred = []

for images, labels in val_dataset:
    predictions = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))
