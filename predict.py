import sys
import joblib
import json
import numpy as np
from collections import Counter

# Load the models
model_names = [
    'temperature_model', 
    'vibration_model', 
    'magnetic_flux_model',
    'audible_sound_model',
    'ultra_sound_model'
]
models = {name: joblib.load(f"{name}.pkl") for name in model_names}

# Define the feature sets used by each model
feature_sets = {
    'temperature_model': ['temperature_one', 'temperature_two'],
    'vibration_model': ['vibration_x', 'vibration_y', 'vibration_z'],
    'magnetic_flux_model': ['magnetic_flux_x', 'magnetic_flux_y', 'magnetic_flux_z'],
    'audible_sound_model': ['vibration_x', 'vibration_y', 'vibration_z', 'audible_sound'],
    'ultra_sound_model': ['vibration_x', 'vibration_y', 'vibration_z', 'ultra_sound']
}

def aggregate_predictions(predictions_list):
    """
    Aggregate multiple predictions into a single result using majority voting for
    categorical predictions and mean for numerical predictions.
    """
    aggregated = {}
    
    # Get all predictions for each model
    for model_name in model_names:
        key = model_name.replace('_model', '')
        model_predictions = [pred[key] for pred in predictions_list]
        
        # Check if predictions are numerical or categorical
        if all(isinstance(x, (int, float)) for x in model_predictions):
            # For numerical predictions, use mean
            aggregated[key] = float(np.mean(model_predictions))
        else:
            # For categorical predictions, use majority vote
            most_common = Counter(model_predictions).most_common(1)[0][0]
            aggregated[key] = str(most_common)
    
    return aggregated

def predict_from_models(input_data_array):
    # Get individual predictions for each input
    all_predictions = []
    
    for input_data in input_data_array:
        predictions = {}
        
        for model_name in model_names:
            features = feature_sets[model_name]
            model = models[model_name]
            
            try:
                # Convert input features to float
                X_input = [float(input_data[feature]) for feature in features]
                prediction = model.predict([X_input])[0]
                
                # Handle both numeric and string predictions
                if isinstance(prediction, (np.integer, np.floating)):
                    prediction = float(prediction)
                else:
                    prediction = str(prediction)
                    
                predictions[model_name.replace('_model', '')] = prediction
                
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid input for {model_name}: {str(e)}")
            
        all_predictions.append(predictions)
    
    # Aggregate all predictions into a single result
    return aggregate_predictions(all_predictions)

# Receive JSON input
input_json = sys.stdin.read()
input_data_array = json.loads(input_json)

# Validate input length
if not isinstance(input_data_array, list):
    print(json.dumps({"error": "Input must be an array"}))
    sys.exit(1)

if len(input_data_array) > 1800:
    print(json.dumps({"error": "Input array exceeds maximum length of 1800"}))
    sys.exit(1)

if len(input_data_array) == 0:
    print(json.dumps({"error": "Input array cannot be empty"}))
    sys.exit(1)

try:
    # Generate aggregated prediction
    prediction = predict_from_models(input_data_array)
    
    # Output the single prediction result as JSON
    print(json.dumps(prediction))
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(1)