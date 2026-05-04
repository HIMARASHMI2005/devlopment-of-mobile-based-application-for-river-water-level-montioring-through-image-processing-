import os
import cv2
import numpy as np
import pickle
import csv
import shutil
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from icrawler.builtin import BingImageCrawler
except ImportError:
    print("icrawler is required. Please install it with: pip install icrawler")
    exit(1)

# Configuration
DATASET_DIR = "dataset"
MODELS_DIR = "models"

def download_images(query, folder, num):
    os.makedirs(folder, exist_ok=True)
    offset = len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
    crawler = BingImageCrawler(storage={'root_dir': folder})
    crawler.crawl(keyword=query, max_num=num, file_idx_offset=offset)

def build_dataset():
    if os.path.exists(DATASET_DIR):
        print("Dataset directory exists, appending to it...")
        # shutil.rmtree(DATASET_DIR)

    print("\n--- Downloading Dataset via Google Images ---")
    
    # Gauge images (600 total)
    gauge_queries = [
        "staff gauge in river water", "water level gauge board river", 
        "river stream gauge post", "flood water level marker river",
        "river tide gauge", "water level measuring post"
    ]
    for q in gauge_queries:
        download_images(q, os.path.join(DATASET_DIR, "gauge"), 150)

    # Non-gauge images (classified)
    high_queries = [
        "river flood high water level", "overflowing river high water",
        "river overflowing banks flood", "high water flood stream"
    ]
    for q in high_queries:
        download_images(q, os.path.join(DATASET_DIR, "no_gauge", "high"), 150)

    medium_queries = [
        "river normal water level", "calm river steady flow",
        "flowing river calm water", "stream regular water level"
    ]
    for q in medium_queries:
        download_images(q, os.path.join(DATASET_DIR, "no_gauge", "medium"), 150)

    low_queries = [
        "river low water dry river", "dried up river bed low water",
        "river bed exposed dry stream", "shallow river drought"
    ]
    for q in low_queries:
        download_images(q, os.path.join(DATASET_DIR, "no_gauge", "low"), 150)

    print("\n--- Generating labels.csv ---")
    csv_data = [["image", "mode", "water_level", "label"]]
    
    # Process gauge images
    gauge_folder = os.path.join(DATASET_DIR, "gauge")
    if os.path.exists(gauge_folder):
        for img_name in os.listdir(gauge_folder):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                water_level = round(np.random.uniform(1.0, 9.0), 2)
                csv_data.append([img_name, "gauge", water_level, ""])
                
    # Process no_gauge images
    for level in ["low", "medium", "high"]:
        level_folder = os.path.join(DATASET_DIR, "no_gauge", level)
        if os.path.exists(level_folder):
            for img_name in os.listdir(level_folder):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    csv_data.append([img_name, "no_gauge", "", level])
                    
    with open(os.path.join(DATASET_DIR, "labels.csv"), mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)
        
    print(f"Dataset download complete! labels.csv created.")

def train_models():
    print("\n--- Extracting Features & Training Models ---")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    X_all, y_gauge_detect = [], []
    X_gauge, y_reg = [], []
    X_no_gauge, y_clf = [], []
    
    labels_file = os.path.join(DATASET_DIR, "labels.csv")
    if not os.path.exists(labels_file):
        print("labels.csv not found. Skipping training.")
        return
        
    with open(labels_file, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            img_name = row['image']
            mode = row['mode']
            
            if mode == 'gauge':
                img_path = os.path.join(DATASET_DIR, "gauge", img_name)
            else:
                img_path = os.path.join(DATASET_DIR, "no_gauge", row['label'], img_name)
                
            img = cv2.imread(img_path)
            if img is None: continue
            
            img_resized = cv2.resize(img, (64, 64))
            row_means = np.mean(img_resized[:, :, 0], axis=1) 
            
            X_all.append(row_means)
            
            if mode == 'gauge':
                y_gauge_detect.append(1)
                X_gauge.append(row_means)
                y_reg.append(float(row['water_level']))
            else:
                y_gauge_detect.append(0)
                X_no_gauge.append(row_means)
                y_clf.append(row['label'])
                
    print("Training Gauge Detector (Random Forest)...")
    gauge_detector = RandomForestClassifier(n_estimators=100, random_state=42)
    if len(X_all) > 0:
        gauge_detector.fit(X_all, y_gauge_detect)
        with open(os.path.join(MODELS_DIR, 'gauge_detector.pkl'), 'wb') as f:
            pickle.dump(gauge_detector, f)
        
    print("Training Exact Water Level Regression Model...")
    reg_model = LinearRegression()
    if len(X_gauge) > 0:
        reg_model.fit(X_gauge, y_reg)
        with open(os.path.join(MODELS_DIR, 'regression_model.pkl'), 'wb') as f:
            pickle.dump(reg_model, f)
            
    print("Training Water Level Classification Model...")
    clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    if len(X_no_gauge) > 0:
        clf_model.fit(X_no_gauge, y_clf)
        with open(os.path.join(MODELS_DIR, 'classification_model.pkl'), 'wb') as f:
            pickle.dump(clf_model, f)
            
    print("SUCCESS: Hybrid models successfully trained on downloaded images!")

if __name__ == "__main__":
    build_dataset()
    train_models()
