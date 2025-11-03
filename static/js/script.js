document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('image-upload');
    const imagePreview = document.getElementById('image-preview');
    const predictBtn = document.getElementById('predict-btn');
    const resultDiv = document.getElementById('result');

    let selectedFile = null;

    // Show image preview when a file is selected
    imageUpload.addEventListener('change', () => {
        selectedFile = imageUpload.files[0];
        if (selectedFile) {
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.innerHTML = ''; 
                const img = document.createElement('img');
                img.src = e.target.result;
                imagePreview.appendChild(img);
            };
            reader.readAsDataURL(selectedFile);
            resultDiv.textContent = '';
        }
    });

    // Handle predict button click
    predictBtn.addEventListener('click', () => {
        if (!selectedFile) {
            // Use a simple, non-blocking message instead of alert()
            resultDiv.textContent = 'Please choose an image first!';
            return;
        }

        const formData = new FormData();
        formData.append('file', selectedFile);

        resultDiv.textContent = 'Analyzing...';

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                resultDiv.textContent = `Error: ${data.error}`;
            } else {
                // --- THIS IS THE FIX ---
                // 1. Look for 'data.category' (which your Flask app sends)
                const prediction = data.category || "Unknown";
                
                // 2. Your Flask app already formats confidence as a string (e.g., "96.30%").
                //    We just need to display it directly.
                const confidence = data.confidence || "N/A";

                // 3. (NEW) Display the extra info from your Flask app
                const emoji = data.emoji || "❓";
                const suggestion = data.suggestion || "No suggestion available.";
                
                // --- Updated Display ---
                // This now shows all the info from your backend.
                resultDiv.innerHTML = `
                    <div class="result-prediction">${emoji} <strong>Prediction:</strong> ${prediction}</div>
                    <div class="result-confidence"><strong>Confidence:</strong> ${confidence}</div>
                    <hr style="margin: 12px 0;">
                    <div class="result-suggestion"><strong>Suggestion:</strong> ${suggestion}</div>
                `;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            resultDiv.textContent = 'An error occurred during prediction.';
        });
    });
});

