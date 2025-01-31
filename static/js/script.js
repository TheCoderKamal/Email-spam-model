// static/js/script.js
document.getElementById('predict-form').addEventListener('submit', async function (e) {
    e.preventDefault();  // Prevent default form submission
    const message = document.getElementById('message').value;  // Get message from textarea

    // Send data to the Flask server
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message })  // Send message as JSON
    });

    const result = await response.json();  // Get JSON response

    // Display the result
    document.getElementById('result').innerText = `Prediction: ${result.prediction}`;
});
