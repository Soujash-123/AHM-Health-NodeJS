// app.js
const { PythonShell } = require('python-shell');
const path = require('path');

// Function to generate random number within a range
function getRandomNumber(min, max, nullable = false) {
    if (nullable && Math.random() < 0.1) { // 10% chance of null value for testing
        return null;
    }
    return parseFloat((Math.random() * (max - min) + min).toFixed(2));
}

// Function to generate random test data
function generateRandomTestData(numSamples = 3) {
    const testData = [];
    
    for (let i = 0; i < numSamples; i++) {
        testData.push({
            temperature_one: getRandomNumber(70, 130, true),      // Temperature range 70-130°C
            temperature_two: getRandomNumber(70, 130, true),      // Temperature range 70-130°C
            vibration_x: getRandomNumber(0.5, 8.0, true),        // Vibration range 0.5-8.0
            vibration_y: getRandomNumber(0.5, 8.0, true),        // Vibration range 0.5-8.0
            vibration_z: getRandomNumber(0.5, 8.0, true),        // Vibration range 0.5-8.0
            magnetic_flux_x: getRandomNumber(0.1, 2.0, true),    // Magnetic flux range 0.1-2.0
            magnetic_flux_y: getRandomNumber(0.1, 2.0, true),    // Magnetic flux range 0.1-2.0
            magnetic_flux_z: getRandomNumber(0.1, 2.0, true),    // Magnetic flux range 0.1-2.0
            audible_sound: getRandomNumber(0.1, 1.0, true),      // Audible sound range 0.1-1.0
            ultra_sound: getRandomNumber(0.1, 1.0, true)         // Ultra sound range 0.1-1.0
        });
    }
    
    return testData;
}

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

// Generate random test data (you can modify the number of samples)
const numSamples = 5; // Generate 5 random samples
const inputDataArray = generateRandomTestData(numSamples);

// Log the generated test data
console.log("Generated Test Data:");
console.log(JSON.stringify(inputDataArray, null, 2));

// Call the prediction function and print the result
predictFromModels(inputDataArray)
    .then(result => {
        console.log("\nPredictions (including overall health):", result.predictions);
        console.log("\nComplete Health Analysis:", result.complete_health_analysis);
        console.log("\nData Quality:", result.data_quality);
    })
    .catch(error => {
        console.error("Error:", error);
    });
