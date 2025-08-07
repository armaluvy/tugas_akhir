import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from transformers import BertForSequenceClassification, AdamW
from sklearn.metrics import classification_report
from tqdm import tqdm

# === 1. Setup GPU dan path ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device digunakan: {device}")

DATASET_PATH = "dataset_3z_ready.pkl"  # file PKL
MODEL_PATH = "indobert-finetuned"        # folder berisi model hasil fine-tune
OUTPUT_DIR = "outputs"
NUM_EPOCHS = 2
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_FOLDS = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 2. Dataset Kustom ===
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

# === 3. Load data dari file PKL ===
df = pd.read_pickle(DATASET_PATH)
input_ids = df['input_ids'].tolist()
attention_mask = df['attention_mask'].tolist()
labels = df['labels'].tolist()
print(len(input_ids), len(attention_mask), len(labels))

assert all(isinstance(x, list) for x in input_ids), "Pastikan input_ids sudah berupa list"
assert all(isinstance(x, list) for x in attention_mask), "Pastikan attention_mask sudah berupa list"

# === 4. K-Fold Cross Validation ===
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
print(f"Jumlah total data: {len(input_ids)}")
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

    model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=3)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # === 5. Training ===
# === 5. Training ===
    model.train()
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in loop:
            batch_input_ids = batch['input_ids'].to(device)
            batch_attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['labels'].to(device)

            outputs = model(input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        labels=batch_labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())


# === 6. Evaluation ===
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch_input_ids = batch['input_ids'].to(device)
            batch_attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['labels'].to(device)

            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())


    report = classification_report(all_labels, all_preds, digits=4)
    print(report)

    with open(os.path.join(OUTPUT_DIR, f"fold_{fold}_report.txt"), "w") as f:
        f.write(report)
model.save_pretrained(f"{OUTPUT_DIR}/fold_{fold}_model")

print("\n✅ Semua fold selesai!")
