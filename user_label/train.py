# train.py
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from tqdm import tqdm
import json
import numpy as np
import logging
import sys
from config import *

# ================================
# Logging setup
# ================================
logger = logging.getLogger("EpochLogger")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(f"{LOG_DIR}/training.log", mode="w", encoding="utf-8")
formatter = logging.Formatter("%(asctime)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

print(f"Using device: {DEVICE}")
print(f"Targeting {len(CANDIDATE_LABELS)} core labels")

# ================================
# Load and filter data — extract text from cleaned.full_text
# ================================
def load_and_filter_data(json_path):
    if not Path(json_path).exists():
        raise FileNotFoundError(f"File not found: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts, labels = [], []
    dropped_no_full_text = 0
    dropped_no_valid_label = 0

    for item in data:
        full_text = item.get("cleaned", {}).get("full_text", "").strip()
        raw_labels = item.get("labels", [])
        
        if not full_text:
            dropped_no_full_text += 1
            continue

        valid_labels = [lbl for lbl in raw_labels if lbl in CANDIDATE_LABELS]
        if len(valid_labels) == 0:
            dropped_no_valid_label += 1
            continue

        texts.append(full_text)
        labels.append(valid_labels)

    print(f"Loaded {len(texts)} training samples")
    if dropped_no_full_text > 0:
        print(f"Skipped {dropped_no_full_text} samples: missing full_text")
    if dropped_no_valid_label > 0:
        print(f"Skipped {dropped_no_valid_label} samples: no valid labels")

    return texts, labels

# ================================
# Custom Dataset
# ================================
class MedicalTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label_vec = self.labels[idx].copy()

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.FloatTensor(label_vec)
        }

# ================================
# Training function
# ================================
def train():
    print("Loading training data...")
    train_texts, train_raw_labels = load_and_filter_data(TRAIN_DATA_JSON)
    val_texts, val_raw_labels = load_and_filter_data(VAL_DATA_JSON)

    if len(train_texts) == 0 or len(val_texts) == 0:
        raise ValueError("Training or validation data is empty. Please check JSON file structure and field names.")

    # Multi-label binarization
    mlb = MultiLabelBinarizer(classes=CANDIDATE_LABELS)
    y_train_bin = mlb.fit_transform(train_raw_labels).astype(np.float32)
    y_val_bin = mlb.transform(val_raw_labels).astype(np.float32)
    num_labels = len(CANDIDATE_LABELS)

    # Save label mapping
    with open(f"{SAVED_MODEL_DIR}/label_classes.json", "w", encoding="utf-8") as f:
        json.dump(CANDIDATE_LABELS, f, ensure_ascii=False, indent=2)

    # Initialize model
    tokenizer = BertTokenizer.from_pretrained(LOCAL_BERT_MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(
        LOCAL_BERT_MODEL_PATH,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    model.to(DEVICE)

    # Data loaders
    train_dataset = MedicalTextDataset(train_texts, y_train_bin, tokenizer, MAX_LEN)
    val_dataset = MedicalTextDataset(val_texts, y_val_bin, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_micro_f1 = 0.0

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", unit="batch", file=sys.stdout)
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)

        # Validation phase
        model.eval()
        preds, true_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating", leave=False):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.sigmoid(logits).cpu().numpy()
                pred = (probs >= THRESHOLD).astype(int)

                preds.extend(pred)
                true_labels.extend(labels.cpu().numpy())

        y_pred = np.array(preds)
        y_true = np.array(true_labels)
        micro_f1 = f1_score(y_true, y_pred, average="micro")
        macro_f1 = f1_score(y_true, y_pred, average="macro")

        logger.info(f"Epoch {epoch+1} - TrainLoss: {avg_loss:.4f}, ValMicroF1: {micro_f1:.4f}, MacroF1: {macro_f1:.4f}")

        if micro_f1 > best_micro_f1:
            best_micro_f1 = micro_f1
            model.save_pretrained(SAVED_MODEL_DIR)
            tokenizer.save_pretrained(SAVED_MODEL_DIR)
            print(f"Best model saved | Micro F1: {micro_f1:.4f}")

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"   Train Loss: {avg_loss:.4f}")
        print(f"   Val Micro F1: {micro_f1:.4f}, Macro F1: {macro_f1:.4f}\n")

    print(f"Training completed. Best Micro F1: {best_micro_f1:.4f}")

if __name__ == "__main__":
    train()
