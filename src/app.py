import os
import json
import torch
import mlflow.pytorch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mlflow.tracking import MlflowClient
from src.qa.data_utils import SquadPreprocessor

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# --- 1. Define data schemas (Input / Output) ---
class QAInput(BaseModel):
    question: str
    context: str

class QAResponse(BaseModel):
    answer: str

# --- 2. Initialize the application ---
app = FastAPI(
    title="MLOps QA API",
    description="API to serve the Question/Answering model in Production",
    version="1.0.0"
)

# Global variables to store the model and vocabulary once loaded
qa_model = None
preprocessor = SquadPreprocessor()
word2idx = {}
idx2word = {}

# --- 3. Load the model at startup ---
@app.on_event("startup")
def load_model():
    global qa_model, word2idx, idx2word
    
    model_name = "qa_model"
    stage = "Production"
    
    print(f"Attempting to load model '{model_name}' from stage '{stage}'...")
    
    try:
        # Load the PyTorch model using the MLflow model registry URI
        model_uri = f"models:/{model_name}/{stage}"
        qa_model = mlflow.pytorch.load_model(model_uri)
        
        # Set the model to evaluation mode (disables dropout layers)
        qa_model.eval() 
        print("✅ Model successfully loaded from MLflow!")
        
        # --- Loading the vocabulary dictionary ---
        # Initialize the MLflow client to interact with the tracking server
        client = MlflowClient()
        
        # Retrieve the model version currently in the Production stage
        versions = client.get_latest_versions(model_name, stages=[stage])
        if not versions:
            raise ValueError(f"No model found in stage '{stage}'")
            
        # Get the run ID associated with this specific model version
        run_id = versions[0].run_id
        
        # Download the word2idx.json artifact from this specific run
        artifact_path = client.download_artifacts(run_id, "word2idx.json")
        
        # Parse the JSON file into a Python dictionary
        with open(artifact_path, "r", encoding="utf-8") as f:
            word2idx = json.load(f)
            
        # Create the reverse mapping (index to word)
        idx2word = {v: k for k, v in word2idx.items()}
        
        print("✅ word2idx dictionary successfully loaded from MLflow artifacts!")
        
    except Exception as e:
        print(f"❌ Error during loading: {e}")

# --- 4. Prediction route ---
@app.post("/predict", response_model=QAResponse)
def predict(data: QAInput):
    if qa_model is None:
        raise HTTPException(status_code=503, detail="The model is not loaded yet or failed to load.")
        
    # 1. Preprocess the question and context
    q_tokens = preprocessor.ultimate_tokenize(data.question)
    c_tokens = preprocessor.ultimate_tokenize(data.context)
    
    q_words = [t["text"] for t in q_tokens]
    c_words = [t["text"] for t in c_tokens]
    
    # 2. Build the input sequence (Question + <SEP> + Context)
    tokens = q_words + ["<SEP>"] + c_words
    
    # Handle unknown words using the loaded word2idx
    unk_id = word2idx.get("<UNK>", 1)
    input_ids = [word2idx.get(w, unk_id) for w in tokens]
    
    # 3. Convert to PyTorch tensor
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    
    # 4. Inference (Prediction)
    with torch.no_grad():
        start_logits, end_logits = qa_model(input_tensor)
        
    # 5. Get the indices with the highest score
    start_idx = torch.argmax(start_logits, dim=1).item()
    end_idx = torch.argmax(end_logits, dim=1).item()
    
    # 6. Reconstruct the answer
    # Account for the offset caused by the question and <SEP> token
    offset = len(q_words) + 1 
    
    if start_idx >= offset and end_idx >= start_idx and end_idx < len(tokens):
        # Extract the words from the context corresponding to the answer span
        answer_tokens = tokens[start_idx:end_idx+1]
        answer = " ".join(answer_tokens)
    else:
        answer = "Sorry, I couldn't find the answer in the context."
        
    return QAResponse(answer=answer)


# Mount the static directory to serve the HTML file
app.mount("/static", StaticFiles(directory="src/static"), name="static")

# --- 5. Main route (Serves the Web Frontend) ---
@app.get("/")
def read_root():
    return FileResponse("src/static/index.html")