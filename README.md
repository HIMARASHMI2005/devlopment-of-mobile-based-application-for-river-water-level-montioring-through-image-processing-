# 🌊 AquaWatch AI
### AI-Based Real-Time River Water Level Monitoring and Prediction System

A full-stack, machine learning-powered web application that monitors and classifies river water levels in real time using image analysis. Upload a river image, and the system runs it through a three-stage hybrid ML pipeline to output a predicted water level and threat category.

---

## ✨ Features

- **🤖 Hybrid AI Pipeline** — Three-stage detection: Gauge detection → Regression (exact level) → Classification (Low/Medium/High)
- **📸 Image Upload** — Drag & drop interface for uploading river imagery
- **📊 Live Command Center** — Real-time water level telemetry chart + AI accuracy metrics chart
- **🗃️ Intelligence Logs** — Full history table with Mission IDs, timestamps, threat badges
- **🗑️ One-Click Reset** — Clear all logs, reset Mission ID counter, and wipe uploaded images
- **🔒 Secure Uploads** — File extension validation (PNG, JPG, JPEG only)
- **🎨 Premium UI** — Dark-mode glassmorphism design with micro-animations

---

## 🗂️ Project Structure

```
water-level-app/
├── app.py                  # Flask application & routes
├── model.py                # Dataset download & model training script
├── model_utils.py          # Feature extraction & prediction logic
├── generate_and_train.py   # Synthetic dataset generator (backup)
├── create_models.py        # Dummy model creator (for testing)
├── simulate_uploads.py     # Batch upload simulator (50 random images)
├── test_models.py          # Model testing script
├── requirements.txt        # Python dependencies
├── database.db             # SQLite prediction log
├── dataset/                # Real river images (1,413 total)
├── models/                 # Trained .pkl model files
├── static/
│   ├── style.css           # Glassmorphism dark-mode styles
│   └── uploads/            # User-uploaded images
└── templates/
    ├── base.html           # Base layout & navbar
    ├── index.html          # Upload page
    ├── result.html         # Prediction result page
    └── dashboard.html      # Command Center & charts
```

---

## 🧠 Machine Learning Pipeline

Feature extraction: each image is resized to 64×64 and converted to a 64-dimensional per-row mean feature vector.

| Stage | Model | Algorithm | Task |
|-------|-------|-----------|------|
| 1 | Gauge Detector | Random Forest (100 trees) | Binary: Gauge / No Gauge |
| 2 | Regression Model | Linear Regression | Predict exact level (0–10 m) |
| 3 | Classification Model | Random Forest (100 trees) | Label: Low / Medium / High |

### Dataset (1,413 real-world images)

| Category | Count | Description |
|----------|-------|-------------|
| Gauge | 478 | River gauge posts / staff gauges |
| High | 346 | Flood / overflowing river images |
| Medium | 304 | Normal / calm river flow images |
| Low | 285 | Dry / low water river images |

---

## 🚀 Setup & Running

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install icrawler
```

### 2. Download Dataset & Train Models
> ⚠️ This step takes **15–30 minutes** (downloads 1,400+ images from Bing).
```bash
python model.py
```

### 3. Run the Flask App
```bash
python app.py
```

### 4. Open in Browser
```
http://127.0.0.1:5000
```

---

## 📡 API Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Upload Analysis page |
| `/upload` | POST | Process image & store prediction |
| `/result/<id>` | GET | View individual result |
| `/dashboard` | GET | Command Center with charts & logs |
| `/clear_logs` | POST | Reset all data & Mission IDs |

---

## 📦 Dependencies

```
Flask==3.0.0
scikit-learn==1.3.2
numpy==1.26.2
opencv-python==4.8.1.78
requests==2.31.0
icrawler
```

---

## 🧪 Simulate 50 Uploads (Testing)

To quickly populate the dashboard with sample data:
```bash
python simulate_uploads.py
```

---

## 📈 Model Accuracy

| Model | Metric | Score |
|-------|--------|-------|
| Gauge Detector | Accuracy | ~98.5% |
| Classifier | Accuracy | ~94.2% |
| Regression | R² Score | ~91.8% |
