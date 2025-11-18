import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import pickle
import os

def load_data():
    """Load and preprocess the hotel reservations dataset"""
    print("Loading data...")
    df = pd.read_csv('data/Hotel Reservations.csv')
    return df

def preprocess_data(df):
    """Clean and engineer features"""
    print("Preprocessing data...")
    
    # Clean categorical columns
    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
    for col in categorical_columns:
        df[col] = df[col].str.lower().str.replace(' ', '_')
    
    # Create season feature
    conditions = [
        df['arrival_month'].isin([12, 1, 2]),
        df['arrival_month'].isin([3, 4, 5]), 
        df['arrival_month'].isin([6, 7, 8])
    ]
    choices = ['winter', 'spring', 'summer']
    df['season'] = np.select(conditions, choices, default='fall')
    
    # Convert target to binary
    df['booking_status'] = (df['booking_status'] == 'canceled').astype(int)
    
    return df

def prepare_features(df):
    """Select and prepare features for training"""
    # Features selected from EDA
    categorical_features = [
        'no_of_special_requests', 
        'arrival_month', 
        'arrival_year', 
        'season', 
        'market_segment_type'
    ]
    numerical_features = ['avg_price_per_room']
    all_features = numerical_features + categorical_features
    
    X = df[all_features]
    y = df['booking_status']
    
    return X, y, all_features

def train_and_save_model():
    """Train the model and save to disk"""
    # Load and prepare data
    df = load_data()
    df = preprocess_data(df)
    X, y, feature_names = prepare_features(df)
    
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Target distribution: {y.mean():.2%} cancellation rate")
    print(f"Using features: {feature_names}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create model pipeline
    print("Training model...")
    model = make_pipeline(
        DictVectorizer(),
        StandardScaler(with_mean=False),
        LogisticRegression(random_state=42, max_iter=1000)
    )
    
    # Train model
    train_dict = X_train.to_dict(orient='records')
    model.fit(train_dict, y_train)
    
    # Evaluate model
    test_dict = X_test.to_dict(orient='records')
    accuracy = model.score(test_dict, y_test)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Save model and features
    os.makedirs('models', exist_ok=True)
    
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    print("Model saved to: models/model.pkl")
    print("Feature names saved to: models/feature_names.pkl")
    
    return model, accuracy

if __name__ == "__main__":
    model, accuracy = train_and_save_model()
    print(f"Training completed! Final accuracy: {accuracy:.4f}")