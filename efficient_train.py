import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from timm import create_model
from transformers import AutoTokenizer
from pycocotools.coco import COCO
from datetime import datetime
from PIL import Image

# Distributed training imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ------------------- DDP Setup Functions ------------------- #
def setup_distributed():
    dist.init_process_group(backend='nccl')
    
def cleanup_distributed():
    dist.destroy_process_group()

# ------------------- Configuration and Constants ------------------- #
DEFAULT_MAX_SEQ_LENGTH = 64
DEFAULT_EMBED_DIM = 512
DEFAULT_NUM_LAYERS = 8
DEFAULT_NUM_HEADS = 8

# ------------------- Data Preparation ------------------- #
class CocoCaptionDataset(Dataset):
    """Custom COCO dataset that returns image-caption pairs with processing"""
    def __init__(self, root, ann_file, transform=None, max_seq_length=DEFAULT_MAX_SEQ_LENGTH):
        self.coco = COCO(ann_file)
        self.root = root
        self.transform = transform
        self.max_seq_length = max_seq_length
        self.ids = list(self.coco.imgs.keys())
        
        # Initialize tokenizer with special tokens
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        special_tokens = {'additional_special_tokens': ['<start>', '<end>']}
        self.tokenizer.add_special_tokens(special_tokens)
        self.vocab_size = len(self.tokenizer)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        
        # Get random caption from available annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        caption = random.choice(anns)['caption']

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        # Tokenize caption with special tokens
        caption = f"<start> {caption} <end>"
        inputs = self.tokenizer(
            caption,
            padding='max_length',
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors='pt',
        )
        return img, inputs.input_ids.squeeze(0)

class CocoTestDataset(Dataset):
    """COCO test dataset that loads images only (no annotations available)"""
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        # Assumes all files in the directory are images
        self.img_files = sorted(os.listdir(root))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.root, img_file)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, img_file  # Return the filename for reference

# ------------------- Model Architecture ------------------- #
class Encoder(nn.Module):
    """CNN encoder using timm models"""
    def __init__(self, model_name='efficientnet_b3', embed_dim=DEFAULT_EMBED_DIM):
        super().__init__()
        self.backbone = create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool='',
            features_only=False
        )
        
        # Get output channels from backbone
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy)
            in_features = features.shape[1]

        self.projection = nn.Linear(in_features, embed_dim)

    def forward(self, x):
        features = self.backbone(x)  # (batch, channels, height, width)
        batch_size, channels, height, width = features.shape
        features = features.permute(0, 2, 3, 1).reshape(batch_size, -1, channels)
        return self.projection(features)

class Decoder(nn.Module):
    """Transformer decoder with positional embeddings and causal masking"""
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, max_seq_length, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Embedding(max_seq_length, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=False
        )
        self.layers = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.max_seq_length = max_seq_length
        
        # Register causal mask buffer
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((max_seq_length, max_seq_length), float('-inf')), diagonal=1)
        )

    def forward(self, x, memory, tgt_mask=None):
        seq_length = x.size(1)
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0)
        x_emb = self.embedding(x) + self.positional_encoding(positions)
        x_emb = self.dropout(x_emb)
        
        # Reshape for transformer: (seq, batch, features)
        x_emb = x_emb.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        
        # Apply causal mask
        mask = self.causal_mask[:seq_length, :seq_length]
        output = self.layers(
            x_emb, 
            memory,
            tgt_mask=mask
        )
        return self.fc(output.permute(1, 0, 2))

class ImageCaptioningModel(nn.Module):
    """Complete image captioning model"""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions, tgt_mask=None):
        memory = self.encoder(images)
        return self.decoder(captions, memory)

# ------------------- Inference Utility ------------------- #
def generate_caption(model, image, tokenizer, device, max_length=DEFAULT_MAX_SEQ_LENGTH):
    """
    Generate a caption for a single image using greedy decoding.
    Assumes the tokenizer has '<start>' and '<end>' as special tokens.
    """
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0)  # shape: (1, 3, H, W)
        if isinstance(model, DDP):
            memory = model.module.encoder(image)
        else:
            memory = model.encoder(image)
        start_token = tokenizer.convert_tokens_to_ids("<start>")
        end_token = tokenizer.convert_tokens_to_ids("<end>")
        caption_ids = [start_token]
        for _ in range(max_length - 1):
            decoder_input = torch.tensor(caption_ids, device=device).unsqueeze(0)
            if isinstance(model, DDP):
                output = model.module.decoder(decoder_input, memory)
            else:
                output = model.decoder(decoder_input, memory)
            next_token_logits = output[0, -1, :]
            next_token = next_token_logits.argmax().item()
            caption_ids.append(next_token)
            if next_token == end_token:
                break
        caption_text = tokenizer.decode(caption_ids, skip_special_tokens=True)
    return caption_text

# ------------------- Training Utilities ------------------- #
def create_dataloaders(args):
    """Create train/val/test dataloaders with appropriate transforms"""
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_set = CocoCaptionDataset(
        root=args.train_image_dir,
        ann_file=args.train_ann_file,
        transform=train_transform
    )

    val_set = CocoCaptionDataset(
        root=args.val_image_dir,
        ann_file=args.val_ann_file,
        transform=eval_transform
    )

    test_set = CocoTestDataset(
        root=args.test_image_dir,
        transform=eval_transform
    )

    # For distributed training, use DistributedSampler
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,  # For inference, process one image at a time
        shuffle=False,
        num_workers=4
    )

    return train_loader, val_loader, test_loader, train_set.tokenizer, train_set

def train_epoch(model, loader, optimizer, criterion, scaler, scheduler, device, args):
    model.train()
    total_loss = 0.0
    if args.distributed:
        loader.sampler.set_epoch(args.epoch)
    for batch_idx, (images, captions) in enumerate(loader):
        images = images.to(device)
        captions = captions.to(device)

        # Teacher forcing: use shifted captions as decoder input
        decoder_input = captions[:, :-1]
        targets = captions[:, 1:].contiguous()

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=args.use_amp):
            logits = model(images, decoder_input)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        scaler.scale(loss).backward()
        if (batch_idx + 1) % args.grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # Update learning rate
            optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, captions in loader:
            images = images.to(device)
            captions = captions.to(device)
            decoder_input = captions[:, :-1]
            targets = captions[:, 1:].contiguous()
            
            logits = model(images, decoder_input)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            total_loss += loss.item()
    
    return total_loss / len(loader)

def main(args):
    if args.distributed:
        setup_distributed()

    device = torch.device("cuda", args.local_rank) if args.distributed else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create dataloaders and obtain tokenizer and training dataset (for sampler)
    train_loader, val_loader, test_loader, tokenizer, train_set = create_dataloaders(args)

    # Initialize model
    encoder = Encoder(args.model_name, args.embed_dim)
    decoder = Decoder(
        vocab_size=tokenizer.vocab_size + 2,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
        dropout=0.1
    )
    model = ImageCaptioningModel(encoder, decoder).to(device)

    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank])

    # Set up training components
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs * len(train_loader),
        eta_min=1e-6
    )
    best_val_loss = float('inf')
    patience_counter = 0

    # Support resume training
    start_epoch = 0
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        if args.distributed:
            model.module.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', best_val_loss)
        print(f"Resumed training from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        args.epoch = epoch  # Useful for the sampler in distributed training
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        if args.local_rank == 0 or not args.distributed:
            print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, scaler, scheduler, device, args
        )
        val_loss = validate(model, val_loader, criterion, device)
        if args.local_rank == 0 or not args.distributed:
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state': model.module.state_dict() if args.distributed else model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'tokenizer': tokenizer,
                }, os.path.join(args.checkpoint_dir, 'best_model.pth'))
            else:
                patience_counter += 1

            if patience_counter >= args.early_stopping_patience:
                print("Early stopping triggered")
                break

    # Inference on test set
    if args.local_rank == 0 or not args.distributed:
        print("\nGenerating captions on test set images:")
        model.eval()
        for idx, (image, filename) in enumerate(test_loader):
            image = image.to(device).squeeze(0)
            caption = generate_caption(model, image, tokenizer, device)
            print(f"{filename}: {caption}")
            if idx >= 4:
                break

    if args.distributed:
        cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data arguments
    parser.add_argument('--train_image_dir', type=str, required=True)
    parser.add_argument('--train_ann_file', type=str, required=True)
    parser.add_argument('--val_image_dir', type=str, required=True)
    parser.add_argument('--val_ann_file', type=str, required=True)
    parser.add_argument('--test_image_dir', type=str, required=True)  # Test set images only

    # Model arguments
    parser.add_argument('--model_name', type=str, default='efficientnet_b3')
    parser.add_argument('--embed_dim', type=int, default=DEFAULT_EMBED_DIM)
    parser.add_argument('--num_layers', type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument('--num_heads', type=int, default=DEFAULT_NUM_HEADS)

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--checkpoint_dir', type=str, default='/workspace')
    parser.add_argument('--early_stopping_patience', type=int, default=3)

    # Distributed training arguments
    # Accept both --local_rank and --local-rank
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0,
                        help="Local rank. Necessary for using distributed training.")
    parser.add_argument('--distributed', action='store_true', help="Use distributed training")

    # Resume training argument
    parser.add_argument('--resume_checkpoint', type=str, default=None, help="Path to checkpoint to resume training from.")

    args = parser.parse_args()

    # Override local_rank from environment variable if set
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    main(args)
