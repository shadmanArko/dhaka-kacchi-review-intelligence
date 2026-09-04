"""
Phase 7: Fine-tuned multi-head DistilBERT.

One shared DistilBERT encoder + 6 small classification heads (one per
aspect), trained jointly on the LLM weak labels. Evaluated against the
same 319-review human gold set used for the baseline, so results are
directly comparable.

Setup:
    pip install torch transformers scikit-learn tqdm
    python finetune_multihead.py

Runs on Apple Silicon GPU automatically (via PyTorch's MPS backend) if
available, falls back to CPU otherwise. Takes a while - progress bars
show per-epoch timing so you can gauge total runtime after epoch 1.

Input:  data_processed.json, labels_weak.jsonl, gold_labels_clean.jsonl,
        gold_sample.json (all already in your project from earlier phases)
Output: model_multihead/ (saved model + tokenizer), results/finetuned_results.json
"""
import json
import random
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

MODEL_NAME = "distilbert-base-uncased"
ASPECTS = ["food_taste", "service", "price", "portion_size", "authenticity", "ambiance"]
LABELS = ["not_mentioned", "negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
VAL_HOLDOUT = 1000  # weak-labeled examples held out from training for monitoring


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_jsonl(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                out[obj["review_id"]] = obj
    return out


class AspectDataset(Dataset):
    def __init__(self, review_ids, text_by_id, labels_by_id, tokenizer):
        self.review_ids = review_ids
        self.text_by_id = text_by_id
        self.labels_by_id = labels_by_id
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.review_ids)

    def __getitem__(self, idx):
        rid = self.review_ids[idx]
        text = self.text_by_id[rid]
        enc = self.tokenizer(
            text, truncation=True, max_length=MAX_LENGTH,
            padding="max_length", return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        for aspect in ASPECTS:
            label_str = self.labels_by_id[rid][aspect]
            item[f"label_{aspect}"] = torch.tensor(LABEL2ID[label_str], dtype=torch.long)
        return item


class MultiAspectDistilBert(nn.Module):
    def __init__(self, model_name, num_labels, num_aspects_dict):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.2)
        self.heads = nn.ModuleDict({
            aspect: nn.Linear(hidden_size, num_labels) for aspect in num_aspects_dict
        })

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # DistilBERT has no pooler - use the [CLS] token's representation
        pooled = out.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        return {aspect: head(pooled) for aspect, head in self.heads.items()}


def compute_class_weights(labels_by_id, review_ids, aspect):
    counts = Counter(labels_by_id[rid][aspect] for rid in review_ids)
    total = sum(counts.values())
    weights = torch.zeros(len(LABELS))
    for label, idx in LABEL2ID.items():
        count = counts.get(label, 1)  # avoid div-by-zero for unseen classes
        weights[idx] = total / (len(LABELS) * count)
    return weights


def main():
    device = get_device()
    print(f"Using device: {device}")

    with open("data/processed/data_processed.json", encoding="utf-8") as f:
        reviews = json.load(f)
    text_by_id = {r["review_id"]: r["text"] for r in reviews}

    weak = load_jsonl("data/processed/labels_weak.jsonl")
    gold = load_jsonl("data/processed/gold_labels.jsonl")
    gold_ids = set(gold.keys())

    all_train_ids = [rid for rid in weak.keys() if rid not in gold_ids and rid in text_by_id]
    random.shuffle(all_train_ids)
    val_ids = all_train_ids[:VAL_HOLDOUT]
    train_ids = all_train_ids[VAL_HOLDOUT:]
    test_ids = [rid for rid in gold.keys() if rid in text_by_id]

    print(f"Train: {len(train_ids)} | Val (weak, held out): {len(val_ids)} | Test (gold): {len(test_ids)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = AspectDataset(train_ids, text_by_id, weak, tokenizer)
    val_ds = AspectDataset(val_ids, text_by_id, weak, tokenizer)
    test_ds = AspectDataset(test_ids, text_by_id, gold, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = MultiAspectDistilBert(MODEL_NAME, len(LABELS), ASPECTS).to(device)

    # per-aspect class weights, computed once from training data - addresses the
    # severe imbalance that hurt the baseline's `neutral` class specifically
    class_weights = {
        aspect: compute_class_weights(weak, train_ids, aspect).to(device) for aspect in ASPECTS
    }
    loss_fns = {aspect: nn.CrossEntropyLoss(weight=class_weights[aspect]) for aspect in ASPECTS}

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)

            loss = sum(
                loss_fns[aspect](logits[aspect], batch[f"label_{aspect}"].to(device))
                for aspect in ASPECTS
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        avg_train_loss = total_loss / len(train_loader)

        # quick validation pass (on held-out weak labels, just to monitor, not a real metric)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids, attention_mask)
                loss = sum(
                    loss_fns[aspect](logits[aspect], batch[f"label_{aspect}"].to(device))
                    for aspect in ASPECTS
                )
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}")

    # --- Final evaluation against the real human gold set ---
    print("\nEvaluating against gold set...")
    model.eval()
    all_preds = {aspect: [] for aspect in ASPECTS}
    all_true = {aspect: [] for aspect in ASPECTS}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            for aspect in ASPECTS:
                preds = logits[aspect].argmax(dim=-1).cpu().tolist()
                true = batch[f"label_{aspect}"].tolist()
                all_preds[aspect].extend(preds)
                all_true[aspect].extend(true)

    results = {}
    for aspect in ASPECTS:
        y_true = [ID2LABEL[i] for i in all_true[aspect]]
        y_pred = [ID2LABEL[i] for i in all_preds[aspect]]
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        results[aspect] = {"accuracy": acc, "macro_f1": macro_f1}
        print(f"\n=== {aspect} ===")
        print(f"  Accuracy vs gold: {acc*100:.1f}%")
        print(f"  Macro F1: {macro_f1:.3f}")
        print(classification_report(y_true, y_pred, zero_division=0))

    overall_acc = sum(r["accuracy"] for r in results.values()) / len(results)
    overall_f1 = sum(r["macro_f1"] for r in results.values()) / len(results)
    print(f"\n=== Overall (mean across aspects) ===")
    print(f"Mean accuracy: {overall_acc*100:.1f}%")
    print(f"Mean macro F1: {overall_f1:.3f}")

    import os
    os.makedirs("results", exist_ok=True)
    with open("results/finetuned_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {"per_aspect": results, "overall_accuracy": overall_acc, "overall_macro_f1": overall_f1},
            f, indent=2,
        )
    print("\nSaved results/finetuned_results.json")

    os.makedirs("model_multihead", exist_ok=True)
    torch.save(model.state_dict(), "model_multihead/model.pt")
    tokenizer.save_pretrained("model_multihead")
    print("Saved model to model_multihead/")


if __name__ == "__main__":
    main()
