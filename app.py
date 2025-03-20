import os
import logging
import subprocess
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key")

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['image']
    # Get the chosen model from the form (default to "resnet50")
    model_choice = request.form.get('model', 'resnet50')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Choose the subprocess command based on the selected model.
            if model_choice == 'efficientnet':
                # For efficientnet, use efficient_caption.py and its arguments.
                command = ['python', 'efficient_caption.py', '--image_dir', filepath]
            else:  # Assume resnet50 or any other value defaults to ResNet
                # Set your resnet checkpoint path as needed.
                RESNET_CHECKPOINT = 'resnet_best_model.pth'
                # Note: resnet_caption.py expects the argument '--image' (not '--image_dir')
                command = ['python', 'resnet_caption.py', '--image', filepath, '--checkpoint', RESNET_CHECKPOINT]
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Clean up the uploaded file
            os.remove(filepath)
            
            # For the resnet model, only use the last line of output as the caption.
            if model_choice == 'efficientnet':
                caption_text = result.stdout.strip()
            else:
                # Split the output into lines and pick the last nonempty one.
                lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
                caption_text = lines[-1] if lines else ""
                
            return jsonify({
                'success': True,
                'caption': caption_text
            })
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Caption generation failed: {e.stderr}")
            return jsonify({'error': 'Failed to generate caption'}), 500
            
    return jsonify({'error': 'Invalid file type'}), 400

