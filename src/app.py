import os
import json
import torch
import mlflow
import mlflow.pytorch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mlflow.tracking import MlflowClient

from qa.data_utils import SquadPreprocessor


class QAInput(BaseModel):
    question: str
    context: str


class QAResponse(BaseModel):
    answer: str


app = FastAPI(
    title="MLOps QA API ",
    description="API to serve the Question/Answering model in Production",
    version="1.0.0",
)

qa_model = None
preprocessor = SquadPreprocessor()
word2idx = {}
idx2word = {}


@app.on_event("startup")
def load_model():
    global qa_model, word2idx, idx2word

    # Allow overriding from Scalingo env vars
    model_name = os.getenv("MODEL_NAME", "qa_model")
    stage = os.getenv("MODEL_STAGE", "Production")

    # MLflow tracking URI comes from env var on Scalingo
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    print(f"Attempting to load model '{model_name}' from stage '{stage}'...")

    try:
        # 1) Load model from MLflow registry
        model_uri = f"models:/{model_name}/{stage}"
        qa_model = mlflow.pytorch.load_model(model_uri)
        qa_model.eval()
        print("✅ Model successfully loaded from MLflow!")

        # 2) Load vocab artifact from the same run
        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[stage])
        if not versions:
            raise ValueError(f"No model found in stage '{stage}'")

        run_id = versions[0].run_id

        # IMPORTANT: must match what train.py logs as artifact
        # If train.py logs artifacts/vocab.json, keep this:
        artifact_path = client.download_artifacts(run_id, "artifacts/vocab.json")

        with open(artifact_path, "r", encoding="utf-8") as f:
            word2idx = json.load(f)

        idx2word = {v: k for k, v in word2idx.items()}
        print("✅ Vocab dictionary successfully loaded from MLflow artifacts!")

    except Exception as e:
        qa_model = None
        word2idx.clear()
        idx2word.clear()
        print(f"❌ Error during loading: {e}")


@app.post("/predict", response_model=QAResponse)
def predict(data: QAInput):
    if qa_model is None:
        raise HTTPException(status_code=503, detail="The model is not loaded yet or failed to load.")
    if not word2idx:
        raise HTTPException(status_code=503, detail="Vocabulary not loaded.")

    # tokenize
    q_tokens = preprocessor.ultimate_tokenize(data.question)
    c_tokens = preprocessor.ultimate_tokenize(data.context)

    q_words = [t["text"] for t in q_tokens]
    c_words = [t["text"] for t in c_tokens]

    tokens = q_words + ["<SEP>"] + c_words

    unk_id = word2idx.get("<UNK>", 1)
    input_ids = [word2idx.get(w, unk_id) for w in tokens]

    input_tensor = torch.tensor([input_ids], dtype=torch.long)

    with torch.no_grad():
        start_logits, end_logits = qa_model(input_tensor)

    start_idx = torch.argmax(start_logits, dim=1).item()
    end_idx = torch.argmax(end_logits, dim=1).item()

    offset = len(q_words) + 1

    if start_idx >= offset and end_idx >= start_idx and end_idx < len(tokens):
        answer_tokens = tokens[start_idx : end_idx + 1]
        answer = " ".join(answer_tokens)
    else:
        answer = "Sorry, I couldn't find the answer in the context."

    return QAResponse(answer=answer)


@app.get("/")
def read_root():
    return {"status": "API online", "model_loaded": qa_model is not None}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": qa_model is not None,
        "vocab_loaded": bool(word2idx),
        "model_name": os.getenv("MODEL_NAME", "qa_model"),
        "model_stage": os.getenv("MODEL_STAGE", "Production"),
    }
