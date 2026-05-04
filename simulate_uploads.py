import os
import random
import requests
import time

URL = "http://127.0.0.1:5000/upload"
DATASET_DIR = "dataset"

def get_all_images():
    images = []
    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                images.append(os.path.join(root, file))
    return images

def simulate_uploads(count=50):
    images = get_all_images()
    if not images:
        print("No images found in dataset folder.")
        return
    
    selected = random.sample(images, min(count, len(images)))
    print(f"Starting simulation of {len(selected)} uploads...")
    
    for i, img_path in enumerate(selected):
        with open(img_path, 'rb') as f:
            files = {'file': (os.path.basename(img_path), f, 'image/jpeg')}
            try:
                # We use allow_redirects=False because /upload redirects to /result, 
                # and we just want to know if the POST was successful.
                response = requests.post(URL, files=files, allow_redirects=False)
                print(f"[{i+1}/{len(selected)}] Uploaded {os.path.basename(img_path)} - Status: {response.status_code}")
            except Exception as e:
                print(f"Error uploading {img_path}: {e}")
        time.sleep(0.05) 

if __name__ == "__main__":
    simulate_uploads(50)
