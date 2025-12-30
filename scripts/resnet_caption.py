#!/usr/bin/env python
import argparse
import torch
from PIL import Image
import nltk
nltk.download('punkt', quiet=True)

# Import the necessary components from resnet_train.py
from resnet_train import EncoderCNN, DecoderRNN, visualize_attention, CONFIG, Vocabulary
import resnet_train  # To update its global vocab variable

def main():
    parser = argparse.ArgumentParser(description="Generate image caption from a trained model.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device,weights_only=False)

    # Initialize models
    encoder = EncoderCNN().to(device)
    decoder = DecoderRNN().to(device)

    # Load state dictionaries
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

    # Update the global vocabulary from the checkpoint
    resnet_train.vocab = checkpoint['vocab']

    # Generate caption using the provided image path
    caption = visualize_attention(args.image, encoder, decoder, device)
    print(caption)

if __name__ == "__main__":
    main()
