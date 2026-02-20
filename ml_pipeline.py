import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from glob import glob

dataset_path = "data/raw/plantvillage_dataset/color"
image_size = 64

X = []
y = []

class_folders = os.listdir(dataset_path)

for label, folder in enumerate(class_folders):
    folder_path = os.path.join(dataset_path, folder)
    images = glob(os.path.join(folder_path, "*.jpg"))
    
    for img_path in images:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (image_size, image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X.append(img.flatten())
        y.append(label)

X = np.array(X, dtype=np.float32) / 255.0
y = np.array(y)

print("Dataset shape:", X.shape)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# PCA
pca = PCA(n_components=150)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("Explained variance:", np.sum(pca.explained_variance_ratio_))

# SVM
svm = SVC(kernel='rbf')
svm.fit(X_train_pca, y_train)

y_pred = svm.predict(X_test_pca)

print("ML Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("\nRunning 5-Fold Cross Validation...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_index, test_index in kf.split(X):
    X_train_k, X_test_k = X[train_index], X[test_index]
    y_train_k, y_test_k = y[train_index], y[test_index]
    
    pca_k = PCA(n_components=150)
    X_train_k = pca_k.fit_transform(X_train_k)
    X_test_k = pca_k.transform(X_test_k)
    
    svm_k = SVC(kernel='rbf')
    svm_k.fit(X_train_k, y_train_k)
    
    y_pred_k = svm_k.predict(X_test_k)
    scores.append(accuracy_score(y_test_k, y_pred_k))

print("K-Fold Accuracy Mean:", np.mean(scores))
print("K-Fold Accuracy Std:", np.std(scores))

