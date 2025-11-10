# predict_finetuned.py
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import json
from config import *

# Load label list
try:
    with open(f"{SAVED_MODEL_DIR}/label_classes.json", "r", encoding="utf-8") as f:
        LABELS = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError("label_classes.json not found. Please run train.py first.")

class MedTagPredictor:
    def __init__(self, model_dir=SAVED_MODEL_DIR):
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = BertForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.model.to(DEVICE)
        print(f"Model loaded successfully, running on {DEVICE}")

    def predict(self, text, top_k=3):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LEN
        ).to(DEVICE)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

        # Get top-k predictions
        topk_idx = probs.argsort()[-top_k:][::-1]
        results = [
            {"label": LABELS[i], "score": round(float(probs[i]), 4)}
            for i in topk_idx if probs[i] >= 0.05
        ]
        return results

    def predict_with_hierarchy(self, text, top_k=3):
        """Return predicted labels along with their major categories."""
        raw_preds = self.predict(text, top_k=top_k)
        enhanced = []
        for r in raw_preds:
            major = get_major_category(r['label'])
            enhanced.append({
                "label": r['label'],
                "score": r['score'],
                "category": major
            })
        return enhanced

# Test examples
if __name__ == "__main__":
    from config import get_major_category

    predictor = MedTagPredictor()

    test_cases = [
        "我空腹血糖12，是不是得了糖尿病？最近还口渴，总想上厕所。",
        "体重下降很快，饭量大但人越来越瘦，会不会是糖尿病？",
        "血压高压160，低压100，头晕，是高血压吗？需要吃药吗？",
        "经常心慌、心跳快，尤其是紧张的时候，是不是心脏病？",
        "胸口闷得慌，有时候像压了块石头，是不是冠心病？",
        "包皮割完有点痒是怎么回事？以前有包茎，现在伤口不红也不肿。",
        "尿尿时有点刺痛，还有白色分泌物，是不是尿路感染？",
        "阴囊胀痛，摸起来像一团蚯蚓，是不是精索静脉曲张？",
        "勃起困难，心理压力大，是不是勃起功能障碍？",
        "月经一直不规律，有时两个月才来一次，是不是内分泌失调？",
        "白带多，发黄有异味，外阴瘙痒，是阴道炎吗？",
        "同房后出血，体检说宫颈有点糜烂，会是宫颈癌吗？",
        "咳嗽咳痰三个月了，最近有点发热，是不是肺炎？",
        "胃老是隐隐作痛，吃完饭更明显，是不是胃炎？",
        "肚子疼，拉肚子好几天了，吃了东西就拉，是不是肠炎？"
    ]

    for text in test_cases:
        print(f"\nInput: {text}")
        res = predictor.predict_with_hierarchy(text)
        for r in res:
            print(f"    {r['label']} ({r['category']}) | Score: {r['score']}")
