# River Water Level Monitoring System

This system uses a hybrid machine learning approach (regression for predicting exact water level and classification for categorizing the level) to monitor river water levels based on uploaded images of gauges.

## Features
- **Upload Interface**: Submit images of river gauges for processing.
- **Hybrid ML Model**: Uses simulated regression to predict exact water level (in meters) and classification (Normal, Warning, Danger).
- **SQLite Database**: Persists historical predictions and uploaded file metadata.
- **Interactive Dashboard**: Displays a Chart.js graph of water level trends over time, alongside a complete table of past predictions.

## Setup Instructions

1. **Install Dependencies**
   Make sure you have Python installed. Then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Dummy Models**
   The application requires two `.pkl` model files. A script is provided to generate these using dummy data.
   ```bash
   python create_models.py
   ```

3. **Create the Dataset Folder (Optional)**
   The system requirements mentioned a 600-image dataset. You can store your actual training images here:
   ```bash
   mkdir dataset
   ```

4. **Run the Flask App**
   ```bash
   python app.py
   ```

5. **Access the App**
   Open your browser and navigate to: `http://127.0.0.1:5000`
