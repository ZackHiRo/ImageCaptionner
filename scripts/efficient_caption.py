import argparse
import logging
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from efficient_train import Encoder, Decoder, ImageCaptioningModel, generate_caption
import os

# Configuration
MODEL_PATH = 'efficient_best_model.pth'  # Path to your saved model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_SEQ_LENGTH = 64  # Ensure this matches the value used during training

# Image transformation (ensure it matches the preprocessing used during training)
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
special_tokens = {'additional_special_tokens': ['<start>', '<end>']}
tokenizer.add_special_tokens(special_tokens)

# Initialize the model components
encoder = Encoder(model_name='efficientnet_b3', embed_dim=512)
decoder = Decoder(
    vocab_size=len(tokenizer),
    embed_dim=512,
    num_layers=8,
    num_heads=8,
    max_seq_length=MAX_SEQ_LENGTH
)
model = ImageCaptioningModel(encoder, decoder).to(DEVICE)

# Load the trained model weights
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}. Please ensure you have a trained model checkpoint at this location.")


# Add a check for the size of the file
if os.path.getsize(MODEL_PATH) == 0:
    raise ValueError(f"Model file at {MODEL_PATH} is empty. Please check the saved model.")


checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

# Check if the checkpoint has the model_state key
if 'model_state' not in checkpoint:
    raise KeyError("The checkpoint file does not contain the key 'model_state'. Please ensure the model was saved correctly using 'torch.save(model.state_dict(), path)'.")


model.load_state_dict(checkpoint['model_state'])
model.eval()



def caption(image_path):

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).to(DEVICE)

    # Generate caption
    caption1 = generate_caption(model, image, tokenizer, DEVICE, max_length=MAX_SEQ_LENGTH)
    return caption1
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a caption for the provided image.")
    parser.add_argument('--image_dir', type=str, required=True, help="Path to the input image file")
    args = parser.parse_args()

    try:
        result = caption(args.image_dir)
        print(result)
    except Exception as e:
        logging.error(f"Error generating caption: {str(e)}")
        exit(1)