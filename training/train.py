# training/train.py

import argparse
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import mlflow
import mlflow.pytorch

from qa.data_utils import SquadPreprocessor, SquadDataset
from qa.model import QAModel
from eval import evaluate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_squad2_original(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def squad2_to_flat(squad_json: Dict[str, Any], keep_impossible: bool = False) -> List[Dict[str, Any]]:
    """
    Convert original SQuAD v2.0 format to a flat list compatible with SquadPreprocessor.process_raw_data():
      {context, question, answer_text, answer_start}
    We typically drop impossible questions for MVP extractive QA.
    """
    out: List[Dict[str, Any]] = []
    for article in squad_json.get("data", []):
        for para in article.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                is_impossible = bool(qa.get("is_impossible", False))
                if is_impossible and not keep_impossible:
                    continue

                question = qa.get("question", "")
                answers = qa.get("answers", [])

                if (not is_impossible) and answers:
                    a0 = answers[0]
                    answer_text = a0.get("text", "")
                    answer_start = int(a0.get("answer_start", -1))
                else:
                    # Will be dropped by preprocessor anyway
                    answer_text = ""
                    answer_start = -1

                out.append(
                    {
                        "context": context,
                        "question": question,
                        "answer_text": answer_text,
                        "answer_start": answer_start,
                    }
                )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()

    # CI helper
    parser.add_argument("--no_mlflow", action="store_true", help="Disable MLflow logging (useful for CI).")

    parser.add_argument("--train_json", type=str, required=True, help="Path to train-v2.0.json (original SQuAD2 format)")
    parser.add_argument("--val_json", type=str, default="", help="Path to dev-v2.0.json (original SQuAD2 format). If missing, split from train.")

    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_answer_len", type=int, default=30)

    parser.add_argument("--mlflow_experiment", type=str, default="qa_bilstm")
    parser.add_argument("--model_name", type=str, default="qa_model")

    parser.add_argument("--subset_train", type=int, default=20000, help="Use first N flat examples (speed). 0 = all.")
    parser.add_argument("--subset_val", type=int, default=4000, help="Use first N flat examples (speed). 0 = all.")

    args = parser.parse_args()
    use_mlflow = not args.no_mlflow

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MLflow setup (tracking URI can be set via env: MLFLOW_TRACKING_URI)
    if use_mlflow:
        mlflow.set_experiment(args.mlflow_experiment)

    # -------- Load SQuAD2 original + flatten --------
    train_squad = load_squad2_original(args.train_json)
    train_flat = squad2_to_flat(train_squad, keep_impossible=False)

    # If val_json exists, use it. Otherwise split from train.
    if args.val_json and os.path.exists(args.val_json):
        val_squad = load_squad2_original(args.val_json)
        val_flat = squad2_to_flat(val_squad, keep_impossible=False)
    else:
        train_flat, val_flat = train_test_split(
            train_flat,
            test_size=0.1,
            random_state=args.seed,
            shuffle=True,
        )

    if args.subset_train and args.subset_train > 0:
        train_flat = train_flat[: args.subset_train]
    if args.subset_val and args.subset_val > 0:
        val_flat = val_flat[: args.subset_val]

    # -------- Preprocess (tokenize + span align) --------
    pre = SquadPreprocessor()
    train_clean = pre.process_raw_data(train_flat)
    val_clean = pre.process_raw_data(val_flat)

    word2idx = pre.build_vocab(train_clean)  # build vocab on train only

    train_ds = SquadDataset(train_clean, word2idx, max_len=args.max_len)
    val_ds = SquadDataset(val_clean, word2idx, max_len=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # -------- Model --------
    model = QAModel(
        vocab_size=len(word2idx),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        pad_idx=word2idx["<PAD>"],
        dropout_rate=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Always write vocab locally (CI + reproducibility)
    vocab_path = artifacts_dir / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(word2idx, f, ensure_ascii=False, indent=2)

    best_f1 = -1.0
    best_weights_path = artifacts_dir / "best_model_state_dict.pt"
    run_id: str | None = None

    with (mlflow.start_run() if use_mlflow else nullcontext()) as run:
        if use_mlflow:
            run_id = run.info.run_id

            mlflow.log_params(
                {
                    "max_len": args.max_len,
                    "embedding_dim": args.embedding_dim,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "epochs": args.epochs,
                    "seed": args.seed,
                    "max_answer_len": args.max_answer_len,
                    "device": str(device),
                    "vocab_size": len(word2idx),
                    "train_flat_size": len(train_flat),
                    "val_flat_size": len(val_flat),
                    "train_clean_size": len(train_clean),
                    "val_clean_size": len(val_clean),
                }
            )

            mlflow.log_artifact(str(vocab_path))

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                start_pos = batch["start_position"].to(device)
                end_pos = batch["end_position"].to(device)

                start_logits, end_logits = model(input_ids)

                loss_start = ce_loss(start_logits, start_pos)
                loss_end = ce_loss(end_logits, end_pos)
                loss = (loss_start + loss_end) / 2

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                running_loss += float(loss.item())
                n_batches += 1

            train_loss = running_loss / max(1, n_batches)

            # validation
            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                device=device,
                ce_loss=ce_loss,
                max_answer_len=args.max_answer_len,
            )

            if use_mlflow:
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_metrics["val_loss"],
                        "val_em": val_metrics["em"],
                        "val_f1": val_metrics["f1"],
                        "n_eval": val_metrics["n_eval"],
                    },
                    step=epoch,
                )

            # keep best
            if val_metrics["f1"] > best_f1:
                best_f1 = float(val_metrics["f1"])
                torch.save(model.state_dict(), best_weights_path)

    # ALWAYS create metrics.json locally (CI + gates)
    metrics_path = artifacts_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"best_val_f1": best_f1, "run_id": run_id}, f, indent=2)

    # MLflow-only: log artifacts/model + registry
    if use_mlflow:
        mlflow.log_artifact(str(best_weights_path))
        mlflow.log_artifact(str(metrics_path))

        # log full model
        mlflow.pytorch.log_model(model, artifact_path="model")

        # register in Model Registry
        model_uri = f"runs:/{run_id}/model"
        registered = mlflow.register_model(model_uri=model_uri, name=args.model_name)

        print(f"Run ID: {run_id}")
        print(f"Registered model: {registered.name} v{registered.version}")

    print(f"Best val F1: {best_f1:.4f}")
    if not use_mlflow:
        print("MLflow disabled (--no_mlflow). Artifacts saved locally under ./artifacts.")


if __name__ == "__main__":
    main()
