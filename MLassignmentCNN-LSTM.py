import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
from PIL import Image
from datasets import load_dataset
from collections import Counter
import numpy as np
from datasets import load_dataset

dataset = load_dataset(
    "parquet",
    data_files={
        "train": "C:\\Users\\Administrator\\Desktop\\pytorch\\vqa-rad\\data\\train-00000-of-00001-eb8844602202be60.parquet",
        "test": "C:\\Users\\Administrator\\Desktop\\pytorch\\vqa-rad\\data\\test-00000-of-00001-e5bc3d208bb4deeb.parquet"
    }
)
# Load the VQA-RAD dataset from Hugging Face
# dataset = load_dataset("flaviagiammarino/vqa-rad")

# Use the train split for this preliminary experiment
train_ds = dataset['train']

# Convert to Pandas DataFrame for easier manipulation
df = pd.DataFrame(train_ds)

# Filter for closed-ended questions (assuming yes/no answers are closed-ended for simplicity;
# in full VQA-RAD, closed-ended include yes/no and short choices; adjust if full metadata available)
closed_df = df[df['answer'].str.lower().isin(['yes', 'no'])]

# Subsets for preliminary experiment (100 training samples, 50 validation)
# Note: Random seed for reproducibility; may have overlap in small dataset, but for demo
train_df = closed_df.sample(n=100, random_state=42)
val_df = closed_df.sample(n=50, random_state=42)

# Build vocabulary from questions in train and val subsets
all_questions = ' '.join(train_df['question'].str.lower().tolist() + val_df['question'].str.lower().tolist()).split()
word_counts = Counter(all_questions)
vocab = {word: idx + 1 for idx, word in enumerate(word_counts)}  # Index 0 reserved for padding
vocab_size = len(vocab) + 1

# Map answers to indices (for classification; e.g., 0: 'no', 1: 'yes')
unique_answers = list(set(train_df['answer'].tolist() + val_df['answer'].tolist()))
answer_to_idx = {ans: idx for idx, ans in enumerate(unique_answers)}
num_answers = len(unique_answers)  # Likely 2 for yes/no

# Custom PyTorch Dataset for VQA-RAD
class VQARADDataset(Dataset):
    def __init__(self, df, vocab, transform=None, max_len=32):
        self.df = df
        self.vocab = vocab
        self.transform = transform
        self.max_len = max_len
        self.answer_to_idx = answer_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = row['image']  # This is a PIL Image object from the dataset

        if self.transform:
            image = self.transform(image)

        # Tokenize question manually (word-level, using vocab)
        question = row['question'].lower().split()
        q_ids = [self.vocab.get(word, 0) for word in question]  # 0 for unknown (though unlikely here)
        length = len(q_ids)
        if length < self.max_len:
            q_ids += [0] * (self.max_len - length)  # Pad with 0
        else:
            q_ids = q_ids[:self.max_len]
            length = self.max_len
        q_ids = torch.tensor(q_ids, dtype=torch.long)

        answer = torch.tensor(self.answer_to_idx[row['answer']], dtype=torch.long)

        return image, q_ids, length, answer

# Image transformations (standard for ResNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create datasets
train_dataset = VQARADDataset(train_df, vocab, transform)
val_dataset = VQARADDataset(val_df, vocab, transform)

# Custom collate function to sort by sequence length for pack_padded_sequence
def collate_fn(batch):
    # Sort batch by question length (descending)
    batch = sorted(batch, key=lambda x: x[2], reverse=True)
    images, q_ids, lengths, answers = zip(*batch)
    images = torch.stack(images)
    q_ids = torch.stack(q_ids)
    answers = torch.stack(answers)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return images, q_ids, lengths, answers

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# Define the CNN-LSTM Baseline Model
class CNNVQA(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, num_answers=2):
        super().__init__()
        # CNN for image features (ResNet-50, pretrained)
        self.cnn = models.resnet50(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, embed_size)  # Adapt final layer

        # Embedding and LSTM for questions
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

        # Fusion and classifier
        self.fc = nn.Linear(embed_size + hidden_size, num_answers)

    def forward(self, images, questions, lengths):
        # Image features
        img_feats = self.cnn(images)

        # Question features
        q_embeds = self.embed(questions)
        packed = pack_padded_sequence(q_embeds, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (h, _) = self.lstm(packed)  # Get last hidden state

        # Fuse and classify
        fused = torch.cat((img_feats, h.squeeze(0)), dim=1)
        return self.fc(fused)

# Instantiate model
model = CNNVQA(vocab_size=vocab_size, num_answers=num_answers)

# Device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Optimizer and loss (for classification)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
avg_losses = []
# Training loop (5 epochs as in preliminary results)
for epoch in range(5):
    model.train()
    total_loss = 0.0
    for images, questions, lengths, answers in train_loader:
        images = images.to(device)
        questions = questions.to(device)
        lengths = lengths.to(device)
        answers = answers.to(device)

        outputs = model(images, questions, lengths)
        loss = criterion(outputs, answers)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    avg_losses.append(avg_loss)
    print(f'Epoch {epoch + 1}/5, Average Loss: {avg_loss:.4f}')

# Evaluation on validation subset
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, questions, lengths, answers in val_loader:
        images = images.to(device)
        questions = questions.to(device)
        lengths = lengths.to(device)
        answers = answers.to(device)

        outputs = model(images, questions, lengths)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == answers).sum().item()
        total += answers.size(0)

accuracy = (correct / total) * 100 if total > 0 else 0
print(f'Validation Accuracy on Closed-Ended Questions: {accuracy:.2f}%')
plt.figure(figsize=(9, 5))
plt.plot(range(1, len(avg_losses)+1), avg_losses,
         marker='o', color='#1f77b4', linewidth=2, markersize=8)
plt.title('Training Loss – CNN + LSTM on VQA-RAD (yes/no subset)', fontsize=14, pad=12)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Cross-Entropy Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(range(1, len(avg_losses)+1))
plt.tight_layout()