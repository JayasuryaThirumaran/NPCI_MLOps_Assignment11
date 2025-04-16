import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

def train_model(preprocessed_dir='preprocessed_data'):
    # Load data
    X = np.load(f'{preprocessed_dir}/X.npy')
    y = np.load(f'{preprocessed_dir}/y.npy')

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")

    # Save model
    joblib.dump(model, 'trained_model/model.pkl')
    print("Model saved as 'model.pkl'.")

if __name__ == '__main__':
    train_model()
