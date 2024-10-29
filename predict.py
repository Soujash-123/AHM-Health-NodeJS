import sys
import joblib
import json
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime

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

def is_valid_number(value):
    """Check if a value is a valid number and not null."""
    if value is None:
        return False
    try:
        float_val = float(value)
        return not (np.isnan(float_val) or np.isinf(float_val))
    except (ValueError, TypeError):
        return False

def safe_mean(values):
    """Calculate mean of values, handling null values."""
    valid_values = [float(v) for v in values if is_valid_number(v)]
    if not valid_values:
        return None
    return sum(valid_values) / len(valid_values)

def evaluate_machine_condition(temperature, vibration):
    """Evaluate machine condition with null value handling."""
    if temperature is None or vibration is None:
        return "Unknown Condition - Insufficient Data"
    
    if temperature < 80 and vibration < 1.8:
        return "Safe Condition"
    elif temperature < 100 and vibration < 2.8:
        return "Maintain Condition"
    else:
        return "Repair Condition"

def detect_temperature_anomaly(temperature):
    """Detect temperature anomalies with null value handling."""
    if temperature is None:
        return "Unable to analyze temperature - Missing Data"
        
    if temperature < 80:
        return "No significant temperature anomaly detected"
    elif 80 <= temperature < 100:
        return "Moderate Overheating - Check Lubrication"
    elif 100 <= temperature < 120:
        return "Significant Overheating - Possible Misalignment or Bearing Wear"
    else:
        return "Critical Overheating - Immediate Repair Needed"

def detect_vibration_anomaly(vibration):
    """Detect vibration anomalies with null value handling."""
    if vibration is None:
        return "Unable to analyze vibration - Missing Data"
        
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
    """Analyze health with null value handling."""
    # Get temperature values
    temp_values = [input_data.get('temperature_one'), input_data.get('temperature_two')]
    avg_temp = safe_mean([v for v in temp_values if is_valid_number(v)])
    
    # Get vibration values
    vib_values = [
        input_data.get('vibration_x'),
        input_data.get('vibration_y'),
        input_data.get('vibration_z')
    ]
    avg_vibration = safe_mean([v for v in vib_values if is_valid_number(v)])
    
    condition = evaluate_machine_condition(avg_temp, avg_vibration)
    
    # Determine overall health status
    if "Unknown Condition" in condition:
        overall_health = "Unknown"
    else:
        overall_health = "Healthy" if condition == "Safe Condition" else "Unhealthy"
    
    return {
        "complete_analysis": {
            "machine_condition": condition,
            "temperature_analysis": detect_temperature_anomaly(avg_temp),
            "vibration_analysis": detect_vibration_anomaly(avg_vibration),
            "data_completeness": {
                "temperature": "Complete" if avg_temp is not None else "Incomplete",
                "vibration": "Complete" if avg_vibration is not None else "Incomplete"
            },
            "timestamp": datetime.now().isoformat()
        },
        "overall_health": overall_health
    }

def aggregate_predictions(predictions_list):
    """Aggregate predictions with null value handling."""
    aggregated = {}
    
    for model_name in model_names:
        key = model_name.replace('_model', '')
        model_predictions = [pred.get(key) for pred in predictions_list if key in pred]
        
        # Remove None values and invalid numbers
        valid_predictions = [p for p in model_predictions if is_valid_number(p)]
        
        if not valid_predictions:
            aggregated[key] = "Insufficient Data"
            continue
            
        # Check if predictions are numerical or categorical
        if all(isinstance(x, (int, float)) for x in valid_predictions):
            # For numerical predictions, use mean
            aggregated[key] = float(np.mean(valid_predictions))
        else:
            # For categorical predictions, use majority vote
            most_common = Counter(valid_predictions).most_common(1)[0][0]
            aggregated[key] = str(most_common)
    
    return aggregated

def predict_from_models(input_data_array):
    """Generate predictions with null value handling."""
    all_predictions = []
    health_analysis = []
    
    for input_data in input_data_array:
        predictions = {}
        
        for model_name in model_names:
            features = feature_sets[model_name]
            model = models[model_name]
            
            # Check if all required features are present and valid
            feature_values = []
            missing_features = False
            
            # Create a pandas DataFrame with the correct feature names
            feature_dict = {}
            for feature in features:
                value = input_data.get(feature)
                if not is_valid_number(value):
                    missing_features = True
                    break
                feature_dict[feature] = [float(value)]
            
            if missing_features:
                predictions[model_name.replace('_model', '')] = "Insufficient Data"
                continue
                
            try:
                # Create DataFrame with proper feature names
                X = pd.DataFrame(feature_dict)
                prediction = model.predict(X)[0]
                
                # Handle both numeric and string predictions
                if isinstance(prediction, (np.integer, np.floating)):
                    prediction = float(prediction)
                else:
                    prediction = str(prediction)
                    
                predictions[model_name.replace('_model', '')] = prediction
                
            except Exception as e:
                predictions[model_name.replace('_model', '')] = f"Prediction Error: {str(e)}"
        
        health_info = analyze_health(input_data)
        all_predictions.append(predictions)
        health_analysis.append(health_info['complete_analysis'])
    
    # Aggregate predictions
    aggregated_predictions = aggregate_predictions(all_predictions)
    
    # Add overall health to aggregated predictions
    health_statuses = [analysis.get('machine_condition', 'Unknown') for analysis in health_analysis]
    if all(status == "Unknown Condition - Insufficient Data" for status in health_statuses):
        overall_health = "Unknown - Insufficient Data"
    else:
        overall_health = "Unhealthy" if "Unhealthy" in [h.get('overall_health', 'Unknown') for h in health_analysis] else "Healthy"
    
    aggregated_predictions['overall_health'] = overall_health
    
    result = {
        "predictions": aggregated_predictions,
        "complete_health_analysis": health_analysis[0] if len(health_analysis) == 1 else health_analysis,
        "data_quality": {
            "total_records": len(input_data_array),
            "complete_records": sum(1 for p in all_predictions if "Insufficient Data" not in p.values()),
            "incomplete_records": sum(1 for p in all_predictions if "Insufficient Data" in p.values())
        }
    }
    
    return result

# Input validation
def validate_input(input_data_array):
    """Validate input data with detailed error messages."""
    if not isinstance(input_data_array, list):
        return {"error": "Input must be an array"}
    if len(input_data_array) > 1800:
        return {"error": "Input array exceeds maximum length of 1800"}
    if len(input_data_array) == 0:
        return {"error": "Input array cannot be empty"}
    return None

# Main execution
try:
    input_json = sys.stdin.read()
    input_data_array = json.loads(input_json)
    
    # Validate input
    validation_error = validate_input(input_data_array)
    if validation_error:
        print(json.dumps(validation_error))
        sys.exit(1)
    
    # Generate prediction and health analysis
    result = predict_from_models(input_data_array)
    
    # Output the result as JSON
    print(json.dumps(result))
except json.JSONDecodeError as e:
    print(json.dumps({"error": f"Invalid JSON input: {str(e)}"}))
    sys.exit(1)
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(1)
