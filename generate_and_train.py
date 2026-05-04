import os
import cv2
import numpy as np
import pickle
import csv
import shutil
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

# Configuration
DATASET_DIR = "dataset"
MODELS_DIR = "models"
NUM_IMAGES = 10000

# Re-create directory structure cleanly
if os.path.exists(DATASET_DIR):
    shutil.rmtree(DATASET_DIR)

os.makedirs(os.path.join(DATASET_DIR, "gauge"), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "no_gauge", "low"), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "no_gauge", "medium"), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "no_gauge", "high"), exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print(f"Generating {NUM_IMAGES} synthetic water images...")

csv_data = [["image", "mode", "water_level", "label"]]

X_all = []
y_gauge_detect = [] # 1 for gauge, 0 for no_gauge

X_gauge = []
y_reg = []

X_no_gauge = []
y_clf = []

np.random.seed(42)

for i in range(1, NUM_IMAGES + 1):
    img_size = (64, 64)
    img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255
    
    is_gauge = np.random.choice([True, False])
    water_level = np.random.uniform(0, 10.0)
    
    # Calculate pixel height of water
    water_h = int((water_level / 10.0) * img_size[1])
    y_start = img_size[1] - water_h
    
    # Fill water area with blue
    if y_start < img_size[1]:
        img[y_start:, :, 0] = 250
        img[y_start:, :, 1] = 100
        img[y_start:, :, 2] = 50
        
    img_name = f"img{i}.jpg"
        
    if is_gauge:
        # Draw a black line to simulate a gauge
        cv2.line(img, (32, 0), (32, 64), (0, 0, 0), 2)
        
        img_path = os.path.join(DATASET_DIR, "gauge", img_name)
        cv2.imwrite(img_path, img)
        
        csv_data.append([img_name, "gauge", round(water_level, 2), ""])
        
        row_means = np.mean(img[:, :, 0], axis=1)
        X_all.append(row_means)
        y_gauge_detect.append(1)
        
        X_gauge.append(row_means)
        y_reg.append(water_level)
    else:
        label = "low"
        if water_level > 7:
            label = "high"
        elif water_level > 4:
            label = "medium"
            
        img_path = os.path.join(DATASET_DIR, "no_gauge", label, img_name)
        cv2.imwrite(img_path, img)
        
        csv_data.append([img_name, "no_gauge", "", label])
        
        row_means = np.mean(img[:, :, 0], axis=1)
        X_all.append(row_means)
        y_gauge_detect.append(0)
        
        X_no_gauge.append(row_means)
        y_clf.append(label)

# Write labels.csv
with open(os.path.join(DATASET_DIR, "labels.csv"), mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

print("Training Models...")
# 1. Gauge Detector (Classification)
gauge_detector = RandomForestClassifier(n_estimators=50, random_state=42)
gauge_detector.fit(X_all, y_gauge_detect)
with open(os.path.join(MODELS_DIR, 'gauge_detector.pkl'), 'wb') as f:
    pickle.dump(gauge_detector, f)

# 2. Regression Model (for exact water level)
reg_model = LinearRegression()
reg_model.fit(X_gauge, y_reg)
with open(os.path.join(MODELS_DIR, 'regression_model.pkl'), 'wb') as f:
    pickle.dump(reg_model, f)

# 3. Classification Model (for low/medium/high)
clf_model = RandomForestClassifier(n_estimators=50, random_state=42)
clf_model.fit(X_no_gauge, y_clf)
with open(os.path.join(MODELS_DIR, 'classification_model.pkl'), 'wb') as f:
    pickle.dump(clf_model, f)

print("--------------------------------------------------")
print(f"Dataset generated at {DATASET_DIR}")
print("labels.csv created.")
print("Models saved: gauge_detector.pkl, regression_model.pkl, classification_model.pkl")
