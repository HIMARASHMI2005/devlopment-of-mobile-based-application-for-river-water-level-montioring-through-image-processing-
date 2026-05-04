import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import os

def create_dummy_models():
    os.makedirs('models', exist_ok=True)
    
    # Dummy data
    X = np.random.rand(100, 10)  # 10 features extracted from image
    y_reg = np.random.uniform(0, 10, 100)  # Water level in meters
    y_clf = np.where(y_reg > 7, 'Danger', np.where(y_reg > 4, 'Warning', 'Normal'))
    
    # Train regression
    reg = LinearRegression()
    reg.fit(X, y_reg)
    with open('models/regression_model.pkl', 'wb') as f:
        pickle.dump(reg, f)
        
    # Train classification
    clf = RandomForestClassifier()
    clf.fit(X, y_clf)
    with open('models/classification_model.pkl', 'wb') as f:
        pickle.dump(clf, f)

if __name__ == '__main__':
    create_dummy_models()
    print("Dummy models created in 'models' directory.")
