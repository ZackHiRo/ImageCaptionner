document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('uploadForm');
    const imageInput = document.getElementById('imageInput');
    const submitBtn = document.getElementById('submitBtn');
    const spinner = submitBtn.querySelector('.spinner-border');
    const resultSection = document.getElementById('resultSection');
    const previewImage = document.getElementById('previewImage');
    const captionText = document.getElementById('captionText');
    const errorAlert = document.getElementById('errorAlert');
    const uploadApp = document.querySelector('.upload-app');

    // Preview image when selected
    imageInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImage.src = e.target.result;
                resultSection.classList.remove('d-none');
                captionText.textContent = '';
                errorAlert.classList.add('d-none');

                // Add success animation class
                uploadApp.classList.add('file-selected');
            };
            reader.readAsDataURL(file);
        }
    });

    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        const formData = new FormData();
        const file = imageInput.files[0];

        if (!file) {
            showError('Please select an image first.');
            return;
        }

        // Add file and selected model to form data
        formData.append('image', file);
        formData.append('model', document.querySelector('input[name="model"]:checked').value);

        // Show loading state
        setLoading(true);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to generate caption');
            }

            // Display the caption
            captionText.textContent = data.caption;
            resultSection.classList.remove('d-none');
            errorAlert.classList.add('d-none');

        } catch (error) {
            showError(error.message || 'An error occurred while generating the caption');
        } finally {
            setLoading(false);
        }
    });

    function setLoading(isLoading) {
        submitBtn.disabled = isLoading;
        spinner.classList.toggle('d-none', !isLoading);
        submitBtn.textContent = isLoading ? ' Processing...' : 'Generate Caption';
        if (isLoading) {
            submitBtn.prepend(spinner);
        }
    }

    function showError(message) {
        errorAlert.textContent = message;
        errorAlert.classList.remove('d-none');
    }
});