import os
import subprocess
import json
import torch
import nltk
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import matplotlib.pyplot as plt
from torchvision import models
from tqdm import tqdm
import torch.distributed as dist
import argparse

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

# Additional imports for extended metrics
from rouge import Rouge
from pycocoevalcap.cider.cider import Cider

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

# ===========================
# CONFIGURATION
# ===========================
CONFIG = {
    # Paths
    "train_ann": r"B:/!S3/Computer Vision/Project/annotations/captions_train2017.json",
    "val_ann": r"B:/!S3/Computer Vision/Project/annotations/captions_val2017.json",
    "train_img_dir": "images/train2017",
    "val_img_dir": "images/val2017",

    # Model
    "img_size": 224,
    "embed_size": 256,
    "hidden_size": 512,
    "attention_dim": 512,
    "feature_map_size": 14,  # From ResNet feature maps
    "dropout": 0.5,          # Dropout probability added

    # Training
    "batch_size": 176,
    "num_epochs": 30,
    "lr": 0.005,
    "fine_tune_encoder": True,
    "grad_clip": 5.0,

    # Vocabulary
    "vocab_threshold": 5,
    "max_len": 20,

    # Beam search
    "beam_size": 3
}

# ===========================
# Vocabulary Builder
# ===========================
class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def build(self, coco, threshold):
        counter = Counter()
        ids = list(coco.anns.keys())
        for ann_id in tqdm(ids):
            caption = coco.anns[ann_id]['caption']
            tokens = nltk.word_tokenize(caption.lower())
            counter.update(tokens)
        # Add special tokens
        self.add_word('<pad>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<unk>')
        # Add words meeting threshold
        for word, cnt in counter.items():
            if cnt >= threshold:
                self.add_word(word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

# Initialize vocab with full training data
coco_train = COCO(CONFIG['train_ann'])
vocab = Vocabulary()
vocab.build(coco_train, CONFIG['vocab_threshold'])
print(f"Vocabulary size: {len(vocab.word2idx)}")


# ===========================
# Attention-based Model
# ===========================
class EncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Use the new weights parameter instead of the deprecated 'pretrained'
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.IMAGENET1K_V1
        resnet = resnet50(weights=weights)
        modules = list(resnet.children())[:-2]
        self.cnn = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((CONFIG['feature_map_size'], CONFIG['feature_map_size']))
        if not CONFIG['fine_tune_encoder']:
            for param in self.cnn.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.cnn(x)  # (batch, 2048, H, W)
        features = self.adaptive_pool(features)  # (batch, 2048, 14, 14)
        features = features.permute(0, 2, 3, 1)   # (batch, 14, 14, 2048)
        features = features.view(features.size(0), -1, features.size(-1))  # (batch, 196, 2048)
        return features

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.U = nn.Linear(CONFIG['hidden_size'], CONFIG['attention_dim'])
        self.W = nn.Linear(2048, CONFIG['attention_dim'])
        self.v = nn.Linear(CONFIG['attention_dim'], 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, hidden):
        U_h = self.U(hidden).unsqueeze(1)  # (batch, 1, attention_dim)
        W_s = self.W(features)              # (batch, 196, attention_dim)
        att = self.tanh(W_s + U_h)          # (batch, 196, attention_dim)
        e = self.v(att).squeeze(2)          # (batch, 196)
        alpha = self.softmax(e)             # (batch, 196)
        context = (features * alpha.unsqueeze(2)).sum(dim=1)  # (batch, 2048)
        return context, alpha

class DecoderRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(len(vocab.word2idx), CONFIG['embed_size'])
        self.lstm = nn.LSTM(CONFIG['embed_size'] + 2048,
                            CONFIG['hidden_size'], batch_first=True)
        self.attention = Attention()
        self.fc = nn.Linear(CONFIG['hidden_size'], len(vocab.word2idx))
        self.dropout = nn.Dropout(p=CONFIG['dropout'])

    def forward(self, features, captions, teacher_forcing_ratio=0.5):
        batch_size = features.size(0)
        h, c = self.init_hidden(features)
        seq_length = captions.size(1) - 1
        outputs = torch.zeros(batch_size, seq_length, len(vocab.word2idx)).to(features.device)
        embeddings = self.dropout(self.embed(captions[:, 0]))
        for t in range(seq_length):
            context, alpha = self.attention(features, h.squeeze(0))
            lstm_input = torch.cat([embeddings, context], dim=1).unsqueeze(1)
            out, (h, c) = self.lstm(lstm_input, (h, c))
            out = self.dropout(out)
            output = self.fc(out.squeeze(1))
            outputs[:, t] = output
            use_teacher_forcing = np.random.random() < teacher_forcing_ratio
            if use_teacher_forcing and t < seq_length - 1:
                embeddings = self.dropout(self.embed(captions[:, t+1]))
            else:
                embeddings = self.dropout(self.embed(output.argmax(dim=-1)))
        return outputs

    def init_hidden(self, features):
        h = torch.zeros(1, features.size(0), CONFIG['hidden_size']).to(features.device)
        c = torch.zeros(1, features.size(0), CONFIG['hidden_size']).to(features.device)
        return h, c

# ===========================
# Enhanced Dataset Class
# ===========================
class CocoDataset(Dataset):
    def __init__(self, ann_file, img_dir, vocab, transform=None):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.vocab = vocab
        self.transform = transform or self.default_transform()
        all_ids = list(self.coco.anns.keys())
        valid_ids = []
        for ann_id in all_ids:
            ann = self.coco.anns[ann_id]
            img_id = ann['image_id']
            file_name = self.coco.loadImgs(img_id)[0]['file_name']
            img_path = os.path.join(self.img_dir, file_name)
            if os.path.exists(img_path):
                valid_ids.append(ann_id)
            else:
                print(f"Warning: File {img_path} not found. Skipping annotation id {ann_id}.")
        self.ids = valid_ids

    def default_transform(self):
        return transforms.Compose([
            transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        ann_id = self.ids[idx]
        ann = self.coco.anns[ann_id]
        img_id = ann['image_id']
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        caption = ann['caption']
        tokens = ['<start>'] + nltk.word_tokenize(caption.lower()) + ['<end>']
        caption_ids = [self.vocab.word2idx.get(token, self.vocab.word2idx['<unk>']) for token in tokens]
        caption_ids += [self.vocab.word2idx['<pad>']] * (CONFIG['max_len'] - len(caption_ids))
        caption_ids = caption_ids[:CONFIG['max_len']]
        return img, torch.tensor(caption_ids)

# ===========================
# Distributed Setup Functions
# ===========================
def setup_distributed():
    dist.init_process_group(backend='nccl')

def cleanup_distributed():
    dist.destroy_process_group()

# ===========================
# Training & Evaluation
# ===========================
def evaluate(encoder, decoder, loader, device, criterion, compute_extended=False):
    encoder.eval()
    decoder.eval()
    total_loss = 0
    # Instantiate smoothing function for BLEU score.
    smoothing_fn = SmoothingFunction().method1
    if compute_extended:
        bleu_scores = []
        meteor_scores = []
        rouge = Rouge()
        rouge1_scores = []
        rougeL_scores = []
        cider_scorer = Cider()
        ref_dict = {}
        hyp_dict = {}
        sample_id = 0
        with torch.no_grad():
            for imgs, caps in loader:
                imgs = imgs.to(device)
                caps = caps.to(device)
                features = encoder(imgs)
                outputs = decoder(features, caps, teacher_forcing_ratio=0)
                loss = criterion(outputs.view(-1, len(vocab.word2idx)), caps[:, 1:].reshape(-1))
                total_loss += loss.item()
                for i in range(imgs.size(0)):
                    predicted_ids = beam_search(features[i].unsqueeze(0), decoder, device)
                    predicted_caption = [vocab.idx2word[idx] for idx in predicted_ids
                                         if idx not in [vocab.word2idx['<start>'], vocab.word2idx['<end>'], vocab.word2idx['<pad>']]]
                    reference_ids = caps[i].tolist()
                    reference_caption = [vocab.idx2word[idx] for idx in reference_ids
                                         if idx not in [vocab.word2idx['<start>'], vocab.word2idx['<end>'], vocab.word2idx['<pad>']]]
                    bleu = sentence_bleu([reference_caption], predicted_caption, smoothing_function=smoothing_fn)
                    bleu_scores.append(bleu)
                    meteor = meteor_score([reference_caption], predicted_caption)
                    meteor_scores.append(meteor)
                    pred_str = " ".join(predicted_caption)
                    ref_str = " ".join(reference_caption)
                    rouge_scores = rouge.get_scores(pred_str, ref_str)
                    rouge1_scores.append(rouge_scores[0]['rouge-1']['f'])
                    rougeL_scores.append(rouge_scores[0]['rouge-l']['f'])
                    ref_dict[sample_id] = [ref_str]
                    hyp_dict[sample_id] = [pred_str]
                    sample_id += 1
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
        avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
        avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
        avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0
        cider_score, _ = cider_scorer.compute_score(ref_dict, hyp_dict)
        metrics = {'BLEU': avg_bleu, 'METEOR': avg_meteor,
                   'ROUGE-1': avg_rouge1, 'ROUGE-L': avg_rougeL, 'CIDEr': cider_score}
        if dist.is_initialized() and dist.get_rank() == 0:
            print(f"Extended Metrics: {metrics}")
        return total_loss / len(loader), metrics
    else:
        with torch.no_grad():
            for imgs, caps in loader:
                imgs = imgs.to(device)
                caps = caps.to(device)
                features = encoder(imgs)
                outputs = decoder(features, caps, teacher_forcing_ratio=0)
                loss = criterion(outputs.view(-1, len(vocab.word2idx)), caps[:, 1:].reshape(-1))
                total_loss += loss.item()
        return total_loss / len(loader)

def beam_search(features, decoder, device):
    k = CONFIG['beam_size']
    start_token = vocab.word2idx['<start>']
    h, c = (decoder.module.init_hidden(features) if isinstance(decoder, torch.nn.parallel.DistributedDataParallel)
            else decoder.init_hidden(features))
    sequences = [[[start_token], 0.0, h, c]]
    for _ in range(CONFIG['max_len'] - 1):
        all_candidates = []
        for seq in sequences:
            tokens, score, h, c = seq
            if tokens[-1] == vocab.word2idx['<end>']:
                all_candidates.append(seq)
                continue
            input_tensor = torch.LongTensor([tokens[-1]]).to(device)
            if isinstance(decoder, torch.nn.parallel.DistributedDataParallel):
                context, _ = decoder.module.attention(features, h.squeeze(0))
                emb = decoder.module.embed(input_tensor)
                lstm_input = torch.cat([emb, context], dim=1).unsqueeze(1)
                out, (h, c) = decoder.module.lstm(lstm_input, (h, c))
                output = decoder.module.fc(out.squeeze(1))
            else:
                context, _ = decoder.attention(features, h.squeeze(0))
                emb = decoder.embed(input_tensor)
                lstm_input = torch.cat([emb, context], dim=1).unsqueeze(1)
                out, (h, c) = decoder.lstm(lstm_input, (h, c))
                output = decoder.fc(out.squeeze(1))
            log_probs = torch.log_softmax(output, dim=1)
            top_probs, top_indices = log_probs.topk(k)
            for i in range(k):
                token = top_indices[0][i].item()
                new_score = score + top_probs[0][i].item()
                new_seq = tokens + [token]
                all_candidates.append([new_seq, new_score, h, c])
        ordered = sorted(all_candidates, key=lambda x: x[1] / len(x[0]), reverse=True)
        sequences = ordered[:k]
    return sequences[0][0]

def visualize_attention(image_path, encoder, decoder, device):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        features = encoder(img_tensor)
        caption_ids = beam_search(features, decoder, device)
    caption = [vocab.idx2word[idx] for idx in caption_ids
               if idx not in [vocab.word2idx['<start>'], vocab.word2idx['<end>'], vocab.word2idx['<pad>']]]
    return ' '.join(caption)

def train(distributed=False, local_rank=0, device=torch.device('cpu'), resume_checkpoint=None):
    train_set = CocoDataset(CONFIG['train_ann'], CONFIG['train_img_dir'], vocab)
    val_set = CocoDataset(CONFIG['val_ann'], CONFIG['val_img_dir'], vocab)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set) if distributed else None
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, shuffle=False) if distributed else None
    train_loader = DataLoader(train_set,
                              batch_size=CONFIG['batch_size'],
                              shuffle=(train_sampler is None),
                              sampler=train_sampler,
                              num_workers=8)
    val_loader = DataLoader(val_set,
                            batch_size=CONFIG['batch_size'],
                            sampler=val_sampler,
                            num_workers=8)
    encoder = EncoderCNN().to(device)
    decoder = DecoderRNN().to(device)
    if distributed:
        encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[local_rank], output_device=local_rank)
        decoder = torch.nn.parallel.DistributedDataParallel(decoder, device_ids=[local_rank], output_device=local_rank)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx['<pad>'])
    if CONFIG['fine_tune_encoder']:
        params = list(decoder.parameters()) + list(encoder.parameters())
    else:
        params = list(decoder.parameters())
    optimizer = optim.Adam(params, lr=CONFIG['lr'])
    # Initialize training state variables
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    # Resume from checkpoint if provided
    if resume_checkpoint is not None:
        print(f"Loading checkpoint from {resume_checkpoint}")
        # Allow Vocabulary as a safe global so it can be unpickled
        torch.serialization.add_safe_globals([Vocabulary])
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("Warning: 'optimizer' state not found in checkpoint. Starting with fresh optimizer state.")
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        print(f"Resumed training from epoch {start_epoch}")
    for epoch in range(start_epoch, CONFIG['num_epochs']):
        if distributed:
            train_sampler.set_epoch(epoch)
        encoder.train()
        decoder.train()
        total_loss = 0
        for imgs, caps in tqdm(train_loader):
            imgs = imgs.to(device)
            caps = caps.to(device)
            optimizer.zero_grad()
            features = encoder(imgs)
            outputs = decoder(features, caps)
            loss = criterion(outputs.view(-1, len(vocab.word2idx)),
                             caps[:, 1:].reshape(-1))
            loss.backward()
            if CONFIG['grad_clip'] is not None:
                nn.utils.clip_grad_norm_(decoder.parameters(), CONFIG['grad_clip'])
            optimizer.step()
            total_loss += loss.item()
        if epoch % 5 == 0:
            val_loss, metrics = evaluate(encoder, decoder, val_loader, device, criterion, compute_extended=True)
            if local_rank == 0:
                print(f"Epoch {epoch+1}/{CONFIG['num_epochs']} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
                with open("metrics_log_Resnet.txt", "a") as f:
                    f.write(f"Epoch {epoch+1}: {metrics}\n")
        else:
            val_loss = evaluate(encoder, decoder, val_loader, device, criterion, compute_extended=False)
            if local_rank == 0:
                print(f"Epoch {epoch+1}/{CONFIG['num_epochs']} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
        if local_rank == 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                checkpoint_path = f'caption_model_best_epoch{epoch}.pth'
                torch.save({
                    'epoch': epoch,
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'epochs_without_improvement': epochs_without_improvement,
                    'vocab': vocab,
                    'config': CONFIG
                }, checkpoint_path)
                #upload_files(epoch)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= 3:
                    print("Early stopping triggered.")
                    break

def upload_files(i):
    files = [f"caption_model_best_epoch{i}.pth", "metrics_log_Resnet.txt"]
    for file in files:
        result = subprocess.run(
            ["rclone", "copy", file, "onedrive:/Computer_Viz/"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"{file} uploaded successfully.")
        else:
            print(f"Error during upload of {file}:", result.stderr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
    args = parser.parse_args()
    if args.distributed:
        setup_distributed()
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0
    train(distributed=args.distributed, local_rank=local_rank, device=device, resume_checkpoint=args.resume)
    if args.distributed:
        cleanup_distributed()
