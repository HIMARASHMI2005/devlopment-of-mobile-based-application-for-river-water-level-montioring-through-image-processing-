import pickle
import numpy as np
import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(BASE_DIR, 'models', 'gauge_detector.pkl'), 'rb') as f:
        gauge_detector = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'models', 'regression_model.pkl'), 'rb') as f:
        reg_model = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'models', 'classification_model.pkl'), 'rb') as f:
        clf_model = pickle.load(f)
except Exception as e:
    print("Warning: Could not load models. Please run generate_and_train.py first.", e)
    gauge_detector, reg_model, clf_model = None, None, None

def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros((1, 64))
    img_resized = cv2.resize(img, (64, 64))
    row_means = np.mean(img_resized[:, :, 0], axis=1)
    return row_means.reshape(1, -1)

def predict_water_level(image_path):
    features = process_image(image_path)
    
    if gauge_detector is None:
        return {"error": "Models not loaded"}
    
    # 1. Detect Gauge
    is_gauge = gauge_detector.predict(features)[0] == 1
    
    if is_gauge:
        # 2. Regression
        exact_level = reg_model.predict(features)[0]
        exact_level = max(0.0, min(10.0, exact_level))
        
        # Determine category based on rules for gauge mode
        if exact_level > 7:
            category = 'High'
        elif exact_level > 4:
            category = 'Medium'
        else:
            category = 'Low'
            
        return {
            "mode": "Gauge",
            "level": round(float(exact_level), 2),
            "category": category
        }
    else:
        # 3. Classification
        category = clf_model.predict(features)[0]
        return {
            "mode": "No Gauge",
            "level": "N/A",
            "category": category.capitalize()
        }
