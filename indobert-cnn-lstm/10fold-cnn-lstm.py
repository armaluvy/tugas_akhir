import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from transformers import BertModel

# === 1. Konfigurasi Umum ===
DATA_PATH = "dataset_3z_ready.pkl"
MODEL_PATH = "indobert-finetuned"
OUTPUT_DIR = "outputs_cnn_lstm"
NUM_EPOCHS = 2
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_FOLDS = 10
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 2. Dataset Custom ===
class IndoBERTDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx]),
            'attention_mask': torch.tensor(self.attention_mask[idx]),
            'labels': torch.tensor(self.labels[idx])
        }

# === 3. CNN-LSTM Model ===
class CNNLSTMClassifier(nn.Module):
    def __init__(self, bert_model, hidden_dim=128, num_classes=3, dropout=0.3):
        super(CNNLSTMClassifier, self).__init__()
        self.bert = bert_model
        for param in self.bert.parameters():
            param.requires_grad = False  # Freeze BERT

        self.conv1 = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # Bidirectional LSTM

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # [B, T, H]

        x = hidden_states.permute(0, 2, 1)  # [B, H, T]
        x = self.relu(self.conv1(x))        # [B, C, T]
        x = x.permute(0, 2, 1)              # [B, T, C]

        lstm_out, _ = self.lstm(x)          # [B, T, 2*hidden_dim]
        pooled = torch.mean(lstm_out, dim=1)  # [B, 2*hidden_dim]

        x = self.dropout(pooled)
        return self.fc(x)

# === 4. Load Dataset ===
df = pd.read_pickle(DATA_PATH)
input_ids = df["input_ids"].tolist()
attention_mask = df["attention_mask"].tolist()
labels = df["labels"].astype(int).tolist()

# === 5. K-Fold Cross Validation ===
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

print(f"📊 Jumlah total data: {len(labels)}")
for fold, (train_idx, val_idx) in enumerate(kf.split(input_ids), 1):
    print(f"\n=== Fold {fold} ===")

    X_train_ids = [input_ids[i] for i in train_idx]
    X_train_mask = [attention_mask[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]

    X_val_ids = [input_ids[i] for i in val_idx]
    X_val_mask = [attention_mask[i] for i in val_idx]
    y_val = [labels[i] for i in val_idx]

    train_dataset = IndoBERTDataset(X_train_ids, X_train_mask, y_train)
    val_dataset = IndoBERTDataset(X_val_ids, X_val_mask, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    bert_model = BertModel.from_pretrained(MODEL_PATH)
    model = CNNLSTMClassifier(bert_model).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # === 6. Training ===
    model.train()
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in loop:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbl = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(ids, mask)
            loss = criterion(outputs, lbl)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

    # === 7. Evaluation ===
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbl = batch["labels"].to(DEVICE)

            outputs = model(ids, mask)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())

    report = classification_report(all_labels, all_preds, digits=4)
    print(report)

    with open(f"{OUTPUT_DIR}/fold_{fold}_report.txt", "w") as f:
        f.write(report)

print("\n✅ Semua fold selesai. Hasil disimpan di:", OUTPUT_DIR)
