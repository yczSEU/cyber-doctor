# config.py
import os
from pathlib import Path

# ================================
# Path Configuration
# ================================
ROOT_DIR = Path(__file__).parent

# Model paths
LOCAL_BERT_MODEL_PATH = "./models/chinese-bert-wwm-ext"
SAVED_MODEL_DIR = ROOT_DIR / "saved_model"
os.makedirs(SAVED_MODEL_DIR, exist_ok=True)

# Data paths
DATASET_DIR = ROOT_DIR / "dataset"
TRAIN_DATA_JSON = "./dataset/train_data.json"
VAL_DATA_JSON = "./dataset/val_data.json"
TEST_DATA_JSON = "./dataset/test_data.json"

# Logging
LOG_DIR = ROOT_DIR / "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ================================
# Candidate Labels (Top-50 High-Frequency)
# ================================
CANDIDATE_LABELS = [
    # Top 1-20
    "免疫功能低下", "癫痫", "腹痛", "糖尿病", "发热",
    "高血压", "便秘", "头痛", "腹泻", "贫血",
    "前列腺炎", "头晕", "阴道炎", "早泄", "尿路感染",
    "不孕不育", "病毒性肝炎", "肺炎", "胃炎", "肝硬化",

    # Top 21-50
    "乙型肝炎", "痔", "月经失调", "功能性消化不良", "失眠障碍",
    "宫颈炎", "甲亢", "营养不良", "盆腔炎", "包皮过长",
    "胸闷", "包皮环切术", "勃起功能障碍", "皮炎", "肝癌",
    "湿疹", "宫颈癌", "脑梗死", "肺癌", "子宫肌瘤",
    "腰痛", "哮喘", "冠状动脉粥样硬化性心脏病", "龟头炎", "痛经",
    "乳腺癌", "慢性胃炎", "HPV感染", "甲减", "心悸"
]

# ================================
# Medical Label Hierarchy (for recommendation enhancement)
# ================================
LABEL_HIERARCHY = {
    # Neurological and psychiatric disorders
    "神经系统疾病": [
        "癫痫", "头痛", "头晕", "脑梗死", "脑出血", "脑卒中", "神经衰弱"
    ],
    "精神心理障碍": [
        "失眠障碍", "抑郁症", "焦虑症", "多梦", "精神分裂症"
    ],

    # Cardiovascular diseases
    "心血管疾病": [
        "高血压", "低血压", "冠状动脉粥样硬化性心脏病", "心绞痛",
        "急性心肌梗死", "心律失常", "心力衰竭", "心悸"
    ],

    # Digestive system
    "消化系统疾病": [
        "胃炎", "慢性胃炎", "功能性消化不良", "消化性溃疡", "胃癌",
        "肝硬化", "乙型肝炎", "病毒性肝炎", "非酒精性脂肪肝", "高脂血症",
        "胰腺炎", "胆囊炎", "肠梗阻", "便血", "腹泻", "便秘", "腹痛"
    ],
    "肛肠疾病": [
        "痔", "肛裂", "直肠癌", "结肠癌"
    ],

    # Endocrine and metabolic disorders
    "内分泌代谢病": [
        "糖尿病", "糖尿病足", "甲亢", "甲减", "甲状腺炎", "甲状腺结节",
        "痛风", "营养不良", "肥胖症", "免疫功能低下"
    ],

    # Respiratory diseases
    "呼吸系统疾病": [
        "肺炎", "哮喘", "COPD", "肺结核", "肺癌", "慢性支气管炎"
    ],

    # Hematological and oncological conditions
    "血液系统疾病": ["贫血", "白血病", "淋巴瘤"],
    "恶性肿瘤": [
        "肝癌", "肺癌", "乳腺癌", "宫颈癌", "卵巢癌", "子宫肌瘤", "皮肤癌",
        "梅毒", "淋病", "HIV感染", "性传播感染"
    ],

    # Infectious and immune-related diseases
    "感染性疾病": [
        "发热", "病毒性肝炎", "HPV感染", "HIV感染", "梅毒", "淋病",
        "皮肤真菌感染", "泌尿生殖道感染"
    ],

    # Male health
    "男性生殖健康": [
        "前列腺炎", "前列腺增生", "精索静脉曲张", "睾丸炎", "附睾炎",
        "包皮过长", "包茎", "包皮环切术", "早泄", "勃起功能障碍"
    ],

    # Female and reproductive health
    "女性生殖健康": [
        "阴道炎", "宫颈炎", "盆腔炎", "月经失调", "痛经", "不孕不育",
        "卵巢囊肿", "多囊卵巢综合征", "乳腺增生", "乳腺纤维腺瘤"
    ],

    # Musculoskeletal disorders
    "骨骼肌肉疾病": [
        "腰痛", "颈椎病", "骨质疏松症", "关节炎", "痛风"
    ],

    # Dermatological conditions
    "皮肤病": [
        "湿疹", "皮炎", "荨麻疹", "皮肤真菌感染", "痤疮"
    ]
}

# ================================
# Reverse mapping: fine-grained label -> major category
# ================================
SUB_TO_MAJOR = {}
for major, subs in LABEL_HIERARCHY.items():
    for sub in subs:
        SUB_TO_MAJOR[sub] = major

def get_major_category(label):
    """Return the major category for a given fine-grained label."""
    return SUB_TO_MAJOR.get(label, "其他")

# ================================
# Hyperparameters
# ================================
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
THRESHOLD = 0.5
TOP_K = 3

# Device configuration
DEVICE = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES", "") != "" else "cpu"
