import pandas as pd
import re
from sklearn.model_selection import train_test_split
import os

# ---- サンプルデータ（4クラス）----
data = {
    "query": [
        # Navigational
        "gmail login", "facebook", "youtube", "amazon", "twitter login",
        "reddit", "linkedin", "instagram", "netflix login", "github",
        # Informational
        "how to make sushi", "what is machine learning", "why is the sky blue",
        "how does bitcoin work", "history of the roman empire",
        "symptoms of covid", "how to learn python", "what is inflation",
        "causes of world war 2", "how to meditate",
        # Transactional
        "buy iphone 15", "download spotify", "order pizza online",
        "book flight to tokyo", "buy nike shoes", "subscribe to netflix",
        "download python", "purchase macbook pro", "get uber", "buy concert tickets",
        # Commercial
        "best laptop 2024", "iphone vs samsung", "top running shoes",
        "best credit card for travel", "macbook pro review",
        "cheapest web hosting", "best python course", "top 10 antivirus software",
        "compare airpods vs sony", "best budget smartphone",
    ],
    "label": (
        ["navigational"] * 10 +
        ["informational"] * 10 +
        ["transactional"] * 10 +
        ["commercial"] * 10
    )
}

df = pd.DataFrame(data)

# ---- テキスト前処理 ----
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

df["query_clean"] = df["query"].apply(clean_text)

# ---- ラベルエンコード ----
label_map = {"navigational": 0, "informational": 1, "transactional": 2, "commercial": 3}
df["label_id"] = df["label"].map(label_map)

# ---- 分割・保存 ----
os.makedirs("data", exist_ok=True)
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label_id"])
train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)

print("✅ データ準備完了")
print(f"   train: {len(train)}件 / test: {len(test)}件")
print(df["label"].value_counts())