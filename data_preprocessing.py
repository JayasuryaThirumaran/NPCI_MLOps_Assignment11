import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(input_csv='dataset/churn_modeling.csv', output_dir='preprocessed_data'):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

    # Label encoding
    le_geo = LabelEncoder()
    le_gender = LabelEncoder()
    df['Geography'] = le_geo.fit_transform(df['Geography'])
    df['Gender'] = le_gender.fit_transform(df['Gender'])

    # Features and target
    X = df.drop('Exited', axis=1).values
    y = df['Exited'].values

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save preprocessed data
    np.save(os.path.join(output_dir, 'X.npy'), X_scaled)
    np.save(os.path.join(output_dir, 'y.npy'), y)
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    joblib.dump(le_geo, os.path.join(output_dir, 'le_geo.pkl'))
    joblib.dump(le_gender, os.path.join(output_dir, 'le_gender.pkl'))

    print(f"Data preprocessing complete. Files saved to '{output_dir}'.")

if __name__ == '__main__':
    preprocess_data()
