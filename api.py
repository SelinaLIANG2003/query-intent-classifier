from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import re

# ---- アプリ初期化 ----
app = FastAPI(title="Query Intent Classifier API")

# ---- モデル読み込み ----
model = pickle.load(open("models/xgb_model.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))

label_map = {0: "navigational", 1: "informational", 2: "transactional", 3: "commercial"}

# ---- テキスト前処理 ----
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

# ---- リクエスト/レスポンス型 ----
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    intent: str
    label_id: int

# ---- エンドポイント ----
@app.get("/")
def root():
    return {"message": "Query Intent Classifier API is running 🚀"}

@app.post("/predict", response_model=QueryResponse)
def predict(request: QueryRequest):
    cleaned = clean_text(request.query)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    return QueryResponse(
        query=request.query,
        intent=label_map[int(pred)],
        label_id=int(pred)
    )