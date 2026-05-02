import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os

# ---- 設定 ----
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 32
BATCH_SIZE = 8
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {DEVICE}")

# ---- データ読み込み ----
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# ---- Dataset ----
class QueryDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df["query_clean"].tolist()
        self.labels = df["label_id"].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ---- モデル・トークナイザー ----
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)
model.to(DEVICE)

train_loader = DataLoader(QueryDataset(train_df, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(QueryDataset(test_df, tokenizer),  batch_size=BATCH_SIZE)

# ---- 学習 ----
optimizer = AdamW(model.parameters(), lr=2e-5)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

# ---- 評価 ----
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(batch["label"].tolist())

acc = accuracy_score(all_labels, all_preds)
f1  = f1_score(all_labels, all_preds, average="macro")

print("=" * 40)
print("BERT Fine-tuning 結果")
print("=" * 40)
print(f"Accuracy : {acc:.4f}")
print(f"F1 (macro): {f1:.4f}")
print()
print(classification_report(all_labels, all_preds,
    target_names=["navigational","informational","transactional","commercial"]))

# ---- モデル保存 ----
os.makedirs("models/bert", exist_ok=True)
model.save_pretrained("models/bert")
tokenizer.save_pretrained("models/bert")
print("✅ BERTモデル保存完了 → models/bert/")