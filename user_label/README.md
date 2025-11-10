# 医疗文本多标签分类系统

本项目基于 BERT 模型实现医疗问诊文本的自动标签预测，支持 Top-K 推荐与层级化标签体系。

## 模型选择

- **基础模型**：`chinese-bert-wwm-ext`  
  - 来源：[Hugging Face Model Hub](https://huggingface.co/hfl/chinese-bert-wwm-ext)
  - 特点：全词掩码（Whole Word Masking）中文预训练模型，在中文 NLP 任务中表现优异
- **任务类型**：多标签文本分类（Multi-label Classification）
- **微调方式**：在医疗问诊数据上进行端到端微调
- **输出层**：Sigmoid 激活 + 二元交叉熵损失（BCEWithLogitsLoss）

> 模型权重目前在本地，后续上传至 Hugging Face Hub：  

## 数据集

原始数据来自 Hugging Face 公开医疗数据集：

- **数据来源**：[shibing624/medical](https://huggingface.co/datasets/shibing624/medical)
- **具体文件**：[`train_zh_0.json`](https://huggingface.co/datasets/shibing624/medical/blob/main/finetune/train_zh_0.json)
- **数据内容**：中文医疗问诊对话，包含患者描述与医生诊断标签
- **处理流程**：
  1. 提取 `cleaned.full_text` 作为输入文本
  2. 筛选高频 Top-50 标签作为候选标签集（见 `config.py`）
  3. 按 8:1:1 划分训练集、验证集、测试集

## 快速使用

### 加载预训练模型（推荐）

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "yczSEU/med-chat-user-label"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "我空腹血糖12，是不是得了糖尿病？最近还口渴，总想上厕所。"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.sigmoid(logits)

# 获取 Top-3 预测
topk = torch.topk(probs[0], k=3)
for score, idx in zip(topk.values, topk.indices):
    print(f"标签: {model.config.id2label[idx.item()]}, 置信度: {score:.4f}")
