import os
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from model_utils import predict_water_level

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

DB_PATH = 'database.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            timestamp DATETIME,
            mode TEXT,
            water_level REAL,
            category TEXT
        )
    ''')
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # ML Prediction
        result = predict_water_level(filepath)
        if "error" in result:
            return "Model Error. Please run generate_and_train.py first."
            
        # Save to DB
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        level_val = result['level'] if result['level'] != "N/A" else None
        
        c.execute('''
            INSERT INTO predictions (filename, timestamp, mode, water_level, category)
            VALUES (?, ?, ?, ?, ?)
        ''', (filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), result['mode'], level_val, result['category']))
        conn.commit()
        pred_id = c.lastrowid
        conn.close()
        
        return redirect(url_for('result', id=pred_id))

@app.route('/result/<int:id>')
def result(id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM predictions WHERE id = ?', (id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        return "Result not found", 404
        
    data = {
        'filename': row[1],
        'timestamp': row[2],
        'mode': row[3],
        'water_level': row[4] if row[4] is not None else "N/A",
        'category': row[5]
    }
    return render_template('result.html', data=data)

@app.route('/dashboard')
def dashboard():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM predictions ORDER BY timestamp DESC')
    rows = c.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            'id': row[0],
            'filename': row[1],
            'timestamp': row[2],
            'mode': row[3],
            'water_level': row[4] if row[4] is not None else "N/A",
            'category': row[5]
        })
        
    return render_template('dashboard.html', history=history)

@app.route('/clear_logs', methods=['POST'])
def clear_logs():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM predictions')
    c.execute('DELETE FROM sqlite_sequence WHERE name="predictions"')
    conn.commit()
    conn.close()
    
    # Also clear the uploads folder
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
            
    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)
