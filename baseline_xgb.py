import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb
import pickle
import os

# ---- データ読み込み ----
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# ---- TF-IDF ベクトル化 ----
vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train["query_clean"])
X_test = vectorizer.transform(test["query_clean"])
y_train = train["label_id"]
y_test = test["label_id"]

# ---- XGBoost 学習 ----
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42
)
model.fit(X_train, y_train)

# ---- 評価 ----
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

print("=" * 40)
print("XGBoost Baseline 結果")
print("=" * 40)
print(f"Accuracy : {acc:.4f}")
print(f"F1 (macro): {f1:.4f}")
print()
print(classification_report(y_test, y_pred,
    target_names=["navigational","informational","transactional","commercial"]))

# ---- モデル保存 ----
os.makedirs("models", exist_ok=True)
pickle.dump(model, open("models/xgb_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/tfidf_vectorizer.pkl", "wb"))
print("✅ モデル保存完了 → models/")