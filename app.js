const { PythonShell } = require('python-shell');
const path = require('path');

// Function to predict from models by calling Python script
function predictFromModels(inputDataArray) {
    return new Promise((resolve, reject) => {
        // Validate input array length
        if (!Array.isArray(inputDataArray)) {
            reject(new Error('Input must be an array'));
            return;
        }
        
        if (inputDataArray.length > 1800) {
            reject(new Error('Input array exceeds maximum length of 1800'));
            return;
        }
        if (inputDataArray.length === 0) {
            reject(new Error('Input array cannot be empty'));
            return;
        }

        // Create options for python-shell
        let options = {
            mode: 'text',
            pythonOptions: ['-u'], // get print results in real-time
            scriptPath: path.join(__dirname),
            args: []
        };
        
        // Run Python script and pass the inputData as JSON string
        let pyShell = new PythonShell('predict.py', options);
        pyShell.send(JSON.stringify(inputDataArray));
        
        pyShell.on('message', (message) => {
            try {
                // Parse the returned JSON message
                let result = JSON.parse(message);
                resolve(result);
            } catch (error) {
                reject(`Error parsing prediction result: ${error}`);
            }
        });

        pyShell.on('stderr', (stderr) => {
            console.error('Python error:', stderr);
        });

        pyShell.on('error', (err) => {
            reject(err);
        });

        pyShell.end((err) => {
            if (err) reject(err);
        });
    });
}

// Example input data array
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
    }
];

// Call the prediction function and print the result
predictFromModels(inputDataArray)
    .then(result => {
        console.log("Predictions (including overall health):", result.predictions);
        console.log("Complete Health Analysis:", result.complete_health_analysis);
    })
    .catch(error => {
        console.error("Error:", error);
    });
