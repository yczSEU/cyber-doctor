# split_train_val_test.py
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from config import CANDIDATE_LABELS, DATASET_DIR

# ================================
# 配置路径
# ================================
INPUT_JSON = "/home/admin/workspace/aop_lab/nas_mount/nas_mount_5/ycz/med_chat/output/labeled_data_all.json"

OUTPUT_DIR = DATASET_DIR
TRAIN_JSON = OUTPUT_DIR / "train_data.json"
VAL_JSON = OUTPUT_DIR / "val_data.json"
TEST_JSON = OUTPUT_DIR / "test_data.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ================================
# 主函数
# ================================
def main():
    print("📥 正在加载原始标注数据...")
    if not Path(INPUT_JSON).exists():
        raise FileNotFoundError(f"❌ 找不到文件: {INPUT_JSON}")

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✅ 成功加载 {len(data)} 条样本")

    cleaned_data = []
    dropped_no_text = 0
    dropped_no_valid_label = 0

    for item in data:
        full_text = item.get("cleaned", {}).get("full_text", "").strip()
        raw_labels = item.get("labels", [])
        
        if not full_text:
            dropped_no_text += 1
            continue
            
        # 只保留候选标签中的项
        valid_labels = [lbl for lbl in raw_labels if lbl in CANDIDATE_LABELS]
        if len(valid_labels) == 0:
            dropped_no_valid_label += 1
            continue

        new_item = item.copy()
        new_item["labels"] = valid_labels
        new_item["label_count"] = len(valid_labels)
        cleaned_data.append(new_item)

    print(f"🧹 过滤统计:")
    print(f"   - 无文本被过滤: {dropped_no_text}")
    print(f"   - 无候选标签被过滤: {dropped_no_valid_label}")
    print(f"✅ 最终保留: {len(cleaned_data)} 条有效样本")

    # 划分 train:val:test = 8:1:1
    train_data, temp = train_test_split(cleaned_data, test_size=0.2, random_state=42, shuffle=True)
    val_data, test_data = train_test_split(temp, test_size=0.5, random_state=42, shuffle=True)

    print(f"📊 分布: 训练集={len(train_data)}, 验证集={len(val_data)}, 测试集={len(test_data)}")

    def save_json(data, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ 已保存至: {path}")

    save_json(train_data, TRAIN_JSON)
    save_json(val_data, VAL_JSON)
    save_json(test_data, TEST_JSON)

    # 输出 top 标签统计
    from collections import Counter
    all_labels = [lbl for item in cleaned_data for lbl in item["labels"]]
    cnt = Counter(all_labels).most_common(20)
    print("\n📈 数据集中高频标签 Top-20:")
    for lbl, freq in cnt:
        print(f"   {lbl}: {freq}")

if __name__ == "__main__":
    main()
