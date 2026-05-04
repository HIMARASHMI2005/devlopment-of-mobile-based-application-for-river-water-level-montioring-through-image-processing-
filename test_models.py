import os
import cv2
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report

MODELS_DIR = "models"
NUM_TEST_IMAGES = 500

print("Loading models for testing...")
try:
    with open(os.path.join(MODELS_DIR, 'gauge_detector.pkl'), 'rb') as f:
        gauge_detector = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'regression_model.pkl'), 'rb') as f:
        reg_model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'classification_model.pkl'), 'rb') as f:
        clf_model = pickle.load(f)
except Exception as e:
    print("Error loading models:", e)
    exit()

print(f"Generating {NUM_TEST_IMAGES} new synthetic images specifically for testing...")

np.random.seed(99) # Different seed to ensure entirely new data for test set

X_all = []
y_gauge_detect = []

X_gauge = []
y_reg_true = []

X_no_gauge = []
y_clf_true = []

for i in range(NUM_TEST_IMAGES):
    img_size = (64, 64)
    img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255
    
    is_gauge = np.random.choice([True, False])
    water_level = np.random.uniform(0, 10.0)
    
    water_h = int((water_level / 10.0) * img_size[1])
    y_start = img_size[1] - water_h
    
    if y_start < img_size[1]:
        img[y_start:, :, 0] = 250
        img[y_start:, :, 1] = 100
        img[y_start:, :, 2] = 50
        
    if is_gauge:
        cv2.line(img, (32, 0), (32, 64), (0, 0, 0), 2)
        row_means = np.mean(img[:, :, 0], axis=1)
        X_all.append(row_means)
        y_gauge_detect.append(1)
        X_gauge.append(row_means)
        y_reg_true.append(water_level)
    else:
        label = "low"
        if water_level > 7:
            label = "high"
        elif water_level > 4:
            label = "medium"
            
        row_means = np.mean(img[:, :, 0], axis=1)
        X_all.append(row_means)
        y_gauge_detect.append(0)
        X_no_gauge.append(row_means)
        y_clf_true.append(label)

print("\n========== TEST RESULTS ==========")

# 1. Test Gauge Detector
y_gauge_pred = gauge_detector.predict(X_all)
gauge_acc = accuracy_score(y_gauge_detect, y_gauge_pred)
print(f"[1] Gauge Detector Accuracy: {gauge_acc * 100:.2f}%")

# 2. Test Regression Model
if len(X_gauge) > 0:
    y_reg_pred = reg_model.predict(X_gauge)
    mae = mean_absolute_error(y_reg_true, y_reg_pred)
    print(f"[2] Regression Mean Absolute Error (MAE): {mae:.4f} meters")

# 3. Test Classification Model
if len(X_no_gauge) > 0:
    y_clf_pred = clf_model.predict(X_no_gauge)
    clf_acc = accuracy_score(y_clf_true, y_clf_pred)
    print(f"[3] Classification Accuracy: {clf_acc * 100:.2f}%")
    print("\nClassification Report (No Gauge):")
    print(classification_report(y_clf_true, y_clf_pred))

print("==================================")
