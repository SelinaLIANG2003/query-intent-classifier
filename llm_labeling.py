import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# ---- LLMシミュレーター（ルールベース）----
# ※ OpenAI APIキーがあれば、この関数をAPI呼び出しに差し替え可能
def mock_llm_classify(query: str) -> str:
    query = query.lower()
    nav_keywords = ["login", "facebook", "gmail", "youtube", "twitter",
                    "reddit", "instagram", "netflix", "github", "linkedin", "amazon"]
    trans_keywords = ["buy", "order", "download", "subscribe", "book",
                      "purchase", "get ", "ticket"]
    comm_keywords = ["best", "vs", "review", "compare", "top", "cheapest",
                     "budget", "recommend"]

    if any(k in query for k in nav_keywords):
        return "navigational"
    elif any(k in query for k in trans_keywords):
        return "transactional"
    elif any(k in query for k in comm_keywords):
        return "commercial"
    else:
        return "informational"

# ---- テストデータで評価 ----
test_df = pd.read_csv("data/test.csv")
label_map = {0: "navigational", 1: "informational", 2: "transactional", 3: "commercial"}
label_map_rev = {v: k for k, v in label_map.items()}

test_df["llm_pred"] = test_df["query_clean"].apply(mock_llm_classify)
test_df["llm_pred_id"] = test_df["llm_pred"].map(label_map_rev)

y_true = test_df["label_id"]
y_pred = test_df["llm_pred_id"]

acc = (y_true == y_pred).mean()
print("=" * 40)
print("LLM (Mock) 結果")
print("=" * 40)
print(f"Accuracy: {acc:.4f}")
print()
print(classification_report(y_true, y_pred,
    target_names=["navigational","informational","transactional","commercial"]))

# ---- 3モデル比較表 ----
results = pd.DataFrame({
    "Model":    ["XGBoost", "BERT", "LLM (Mock)"],
    "Accuracy": [0.375,     0.875,  acc],
    "F1_macro": [0.278,     0.867,  None]
})
from sklearn.metrics import f1_score
results.loc[2, "F1_macro"] = f1_score(y_true, y_pred, average="macro")
print()
print(results.to_string(index=False))

# ---- 可視化 ----
os.makedirs("results", exist_ok=True)

# 棒グラフ
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(3)
bars = ax.bar(x, results["Accuracy"], width=0.35, label="Accuracy", color="#4C72B0")
bars2 = ax.bar(x + 0.35, results["F1_macro"], width=0.35, label="F1 Macro", color="#DD8452")
ax.set_xticks(x + 0.175)
ax.set_xticklabels(results["Model"])
ax.set_ylim(0, 1.1)
ax.set_title("Model Comparison: Accuracy & F1")
ax.legend()
plt.tight_layout()
plt.savefig("results/model_comparison.png", dpi=150)
print("✅ 図保存 → results/model_comparison.png")

# 混淆矩阵
fig2, ax2 = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_true, y_pred,
    display_labels=["nav","info","trans","comm"],
    ax=ax2, colorbar=False
)
ax2.set_title("Confusion Matrix (LLM Mock)")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150)
print("✅ 混淆矩阵保存 → results/confusion_matrix.png")