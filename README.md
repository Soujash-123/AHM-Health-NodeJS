# Asset Health Monitoring System - NodeJS + Python

This service processes sensor data through multiple ML models to predict system states. It supports batch processing of up to 1800 inputs and returns an aggregated prediction.

## Prerequisites

- Node.js (v12 or higher)
- Python (v3.6 or higher)
- npm (comes with Node.js)
- pip (Python package manager)

## Installation

### 1. Node.js Dependencies

Install the required Node.js packages:

```bash
npm install python-shell
```

### 2. Python Dependencies

You have two options for installing Python dependencies:

#### Option A: Using Virtual Environment (Recommended for development)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate

# Install dependencies
pip install numpy joblib scikit-learn
```

#### Option B: Global Installation (If pip is supported at root)

If you have root/admin access and prefer a global installation:

```bash
pip install numpy joblib scikit-learn
```

## Project Structure

Ensure your project directory contains these files:

```
project/
├── app.js                    # Node.js application
├── predict.py               # Python prediction script
├── temperature_model.pkl    # ML model
├── vibration_model.pkl      # ML model
├── magnetic_flux_model.pkl  # ML model
├── audible_sound_model.pkl  # ML model
└── ultra_sound_model.pkl    # ML model
```

## Usage

### 1. Input Format

The service accepts an array of sensor readings. Each reading should have the following format:

```javascript
{
    "temperature_one": number,
    "temperature_two": number,
    "vibration_x": number,
    "vibration_y": number,
    "vibration_z": number,
    "magnetic_flux_x": number,
    "magnetic_flux_y": number,
    "magnetic_flux_z": number,
    "audible_sound": number,
    "ultra_sound": number
}
```

### 2. Running the Service

```bash
node app.js
```

### 3. Example Code

```javascript
const inputDataArray = [
    {
        temperature_one: 90,
        temperature_two: 85,
        vibration_x: 2.5,
        vibration_y: 2.0,
        vibration_z: 1.8,
        magnetic_flux_x: 0.9,
        magnetic_flux_y: 1.0,
        magnetic_flux_z: 1.1,
        audible_sound: 0.3,
        ultra_sound: 0.2
    },
    // ... more readings (up to 1800)
];

predictFromModels(inputDataArray)
    .then(prediction => {
        console.log("Prediction:", prediction);
    })
    .catch(error => {
        console.error("Error:", error);
    });
```

### 4. Output Format

The service returns a single prediction object aggregating all inputs:

```javascript
{
    "temperature": "healthy",
    "vibration": "normal",
    "magnetic_flux": "stable",
    "audible_sound": "normal",
    "ultra_sound": "normal"
}
```

## Notes

- Maximum input array length is 1800 entries
- The service aggregates multiple inputs into a single prediction using:
  - Majority voting for categorical predictions
  - Mean values for numerical predictions
- Ensure all model (.pkl) files are present in the project directory
- All sensor readings should be numerical values

## Error Handling

The service provides error messages for:
- Invalid input formats
- Exceeding maximum input length (1800)
- Empty input arrays
- Model prediction errors

## Deactivating Virtual Environment

If using a virtual environment, deactivate it when done:

```bash
deactivate
```
