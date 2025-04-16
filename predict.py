import joblib
import numpy as np

def predict(input_data):
    model = joblib.load('trained_model/model.pkl')
    scaler = joblib.load('preprocessed_data/scaler.pkl')
    le_geo = joblib.load('preprocessed_data/le_geo.pkl')
    le_gender = joblib.load('preprocessed_data/le_gender.pkl')

    # Sample input should be a dictionary
    data = input_data.copy()
    data['Geography'] = le_geo.transform([data['Geography']])[0]
    data['Gender'] = le_gender.transform([data['Gender']])[0]

    features = [
        data['CreditScore'], data['Geography'], data['Gender'],
        data['Age'], data['Tenure'], data['Balance'], data['NumOfProducts'],
        data['HasCrCard'], data['IsActiveMember'], data['EstimatedSalary']
    ]

    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)

    return prediction[0]

if __name__ == '__main__':
    sample_input = {
        'CreditScore': 600,
        'Geography': 'France',
        'Gender': 'Female',
        'Age': 40,
        'Tenure': 3,
        'Balance': 60000.0,
        'NumOfProducts': 2,
        'HasCrCard': 1,
        'IsActiveMember': 1,
        'EstimatedSalary': 50000.0
    }

    result = predict(sample_input)
    print(f"Prediction (Exited=1 / Not Exited=0): {result}")
