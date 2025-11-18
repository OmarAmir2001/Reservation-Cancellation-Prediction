from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Global variables for loaded model
model = None
feature_names = None

def load_model():
    """Load the trained model and feature names"""
    global model, feature_names
    
    try:
        with open('models/model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('models/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        print("Model loaded successfully")
        print(f"Expected features: {feature_names}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def validate_input(data):
    """Validate and prepare input data"""
    # Create DataFrame
    df_input = pd.DataFrame([data])
    
    # Ensure all expected features are present
    for feature in feature_names:
        if feature not in df_input.columns:
            df_input[feature] = 0  # Default for missing features
    
    return df_input[feature_names]

@app.route('/')
def home():
    return jsonify({
        'message': 'Hotel Reservation Cancellation Prediction API',
        'status': 'active',
        'version': '1.0.0',
        'endpoints': {
            'health': 'GET /health',
            'predict': 'POST /predict',
            'batch_predict': 'POST /batch_predict'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy', 
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict_cancellation():
    """Predict cancellation for a single reservation"""
    try:
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate and preprocess input
        df_processed = validate_input(data)
        input_dict = df_processed.to_dict(orient='records')[0]
        
        # Make prediction
        prediction = model.predict([input_dict])[0]
        probability = model.predict_proba([input_dict])[0][1]
        
        # Prepare response
        response = {
            'prediction': int(prediction),
            'probability': float(probability),
            'status': 'canceled' if prediction == 1 else 'not_canceled',
            'confidence': float(probability) if prediction == 1 else float(1 - probability)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict cancellations for multiple reservations"""
    try:
        data = request.get_json()
        
        if not data or 'reservations' not in data:
            return jsonify({'error': 'No reservations provided'}), 400
        
        reservations = data['reservations']
        results = []
        
        for i, reservation in enumerate(reservations):
            try:
                df_processed = validate_input(reservation)
                input_dict = df_processed.to_dict(orient='records')[0]
                
                prediction = model.predict([input_dict])[0]
                probability = model.predict_proba([input_dict])[0][1]
                
                results.append({
                    'reservation_id': i,
                    'prediction': int(prediction),
                    'probability': float(probability),
                    'status': 'canceled' if prediction == 1 else 'not_canceled'
                })
            except Exception as e:
                results.append({
                    'reservation_id': i,
                    'error': str(e)
                })
        
        return jsonify({'predictions': results})
        
    except Exception as e:
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Start Flask server
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=False)