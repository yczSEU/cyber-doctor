# baseline_predict.py
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
import json
from pathlib import Path
from config import *

def load_test_data(json_path):
    if not Path(json_path).exists():
        raise FileNotFoundError(f"Error: Test file not found: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts, labels = [], []
    for item in data:
        full_text = item.get("cleaned", {}).get("full_text", "").strip()
        raw_labels = item.get("labels", [])
        valid_labels = [lbl for lbl in raw_labels if lbl in CANDIDATE_LABELS]
        if full_text and valid_labels:
            texts.append(full_text)
            labels.append(valid_labels)
    print(f"Loaded test set successfully, total {len(texts)} valid samples")
    return texts, labels

def hit_at_k(y_true, y_prob, k=3):
    hit_count = 0
    total_samples = 0
    for i in range(len(y_true)):
        true_idx = np.where(y_true[i] == 1)[0]
        if len(true_idx) == 0:
            continue
        topk_idx = np.argsort(y_prob[i])[-k:]
        if len(np.intersect1d(true_idx, topk_idx)) > 0:
            hit_count += 1
        total_samples += 1
    return hit_count / total_samples if total_samples > 0 else 0

def dcg_at_k(scores, k):
    scores = np.asarray(scores[:k], dtype=np.float64)
    log_positions = np.log2(np.arange(2, len(scores) + 2))
    return np.sum(scores / log_positions)

def ndcg_at_k(y_true, y_prob, k=3):
    ndcg_scores = []
    for i in range(len(y_true)):
        true_binary = y_true[i]
        pred_scores = y_prob[i]
        topk_idx = np.argsort(pred_scores)[-k:][::-1]
        actual_gains = [true_binary[idx] for idx in topk_idx]
        ideal_gains = sorted(true_binary, reverse=True)[:k]
        dcg = dcg_at_k(actual_gains, k)
        idcg = dcg_at_k(ideal_gains, k)
        ndcg_scores.append(dcg / max(idcg, 1e-8))
    return np.mean(ndcg_scores)

def main():
    print("Loading test data...")
    test_texts, test_true_labels = load_test_data(TEST_DATA_JSON)

    label_to_id = {label: i for i, label in enumerate(CANDIDATE_LABELS)}
    num_labels = len(CANDIDATE_LABELS)
    print(f"Total number of labels: {num_labels}")

    tokenizer = BertTokenizer.from_pretrained(LOCAL_BERT_MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(
        LOCAL_BERT_MODEL_PATH,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    model.to(DEVICE)
    model.eval()

    y_true = np.zeros((len(test_true_labels), num_labels))
    for i, labels in enumerate(test_true_labels):
        for lbl in labels:
            if lbl in label_to_id:
                y_true[i][label_to_id[lbl]] = 1

    all_probs = []
    with torch.no_grad():
        for i in range(0, len(test_texts), BATCH_SIZE):
            batch_texts = test_texts[i:i+BATCH_SIZE]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(DEVICE)
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            all_probs.append(probs)
    y_prob = np.vstack(all_probs)

    hit3 = hit_at_k(y_true, y_prob, k=TOP_K)
    ndcg3 = ndcg_at_k(y_true, y_prob, k=TOP_K)

    print("\n" + "="*60)
    print("BERT base model evaluation on test set")
    print("="*60)
    print(f"Model: chinese-bert-wwm-ext (no fine-tuning)")
    print(f"Task: Top-{TOP_K} recommendation")
    print(f"Hit@{TOP_K}: {hit3:.4f}")
    print(f"NDCG@{TOP_K}: {ndcg3:.4f}")
    print("="*60)

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    main()
