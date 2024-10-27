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

def evaluate_machine_condition(temperature, vibration):
    if temperature < 80 and vibration < 1.8:
        return "Safe Condition"
    elif temperature < 100 and vibration < 2.8:
        return "Maintain Condition"
    else:
        return "Repair Condition"

def detect_temperature_anomaly(temperature):
    if temperature < 80:
        return "No significant temperature anomaly detected"
    elif 80 <= temperature < 100:
        return "Moderate Overheating - Check Lubrication"
    elif 100 <= temperature < 120:
        return "Significant Overheating - Possible Misalignment or Bearing Wear"
    else:
        return "Critical Overheating - Immediate Repair Needed"

def detect_vibration_anomaly(vibration):
    if vibration < 1.8:
        return "No significant vibration anomaly detected"
    elif 1.8 <= vibration < 2.8:
        return "Unbalance Fault"
    elif 2.8 <= vibration < 4.5:
        return "Misalignment Fault"
    elif 4.5 <= vibration < 7.1:
        return "Looseness Fault"
    else:
        return "Bearing Fault or Gear Mesh Fault"

def analyze_health(input_data):
    # Calculate average temperature and vibration
    avg_temp = (float(input_data['temperature_one']) + float(input_data['temperature_two'])) / 2
    avg_vibration = (float(input_data['vibration_x']) + float(input_data['vibration_y']) + float(input_data['vibration_z'])) / 3
    
    return {
        "machine_condition": evaluate_machine_condition(avg_temp, avg_vibration),
        "temperature_analysis": detect_temperature_anomaly(avg_temp),
        "vibration_analysis": detect_vibration_anomaly(avg_vibration)
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
    health_analysis = []
    
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
        health_analysis.append(analyze_health(input_data))
    
    # Aggregate predictions and health analysis
    aggregated_predictions = aggregate_predictions(all_predictions)
    aggregated_health = health_analysis[0] if len(health_analysis) == 1 else health_analysis
    
    # Combine predictions with health analysis
    result = {
        "predictions": aggregated_predictions,
        "overall_health": aggregated_health
    }
    
    return result

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
    # Generate prediction and health analysis
    result = predict_from_models(input_data_array)
    
    # Output the result as JSON
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(1)
