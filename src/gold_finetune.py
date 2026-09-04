"""
Gold fine-tuning experiment.

Starts from the model already trained on weak labels (Phase 7), then
continues training on a small slice of REAL human gold labels. Tests
whether the earlier plateau was caused by weak-label noise (as
hypothesized) - if a small dose of real signal meaningfully improves
results, that confirms it.

IMPORTANT: fine-tuning on gold data means we can no longer evaluate on
all 319 gold reviews (that would be testing on training data). The 319
are split three ways instead:
    - gold_train  (~65%): what the model is fine-tuned on
    - gold_val    (~15%): used only to pick the best epoch (early stopping)
    - gold_test   (~20%): held out completely, never touched until the
                          final "before vs after" comparison

This means the final numbers come from a smaller test set (~64 reviews)
than the Phase 6/7 comparisons (319 reviews) - expect noisier per-aspect
numbers, especially for rare classes. That's a real, worth-stating
limitation, not a bug.

Setup: (same dependencies as finetune_multihead.py, already installed)
    python gold_finetune.py

Requires: model_multihead/ from the Phase 7 run (loads its weights as
the starting point).
"""
import json
import random
import os
import copy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, classification_report

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

MODEL_NAME = "distilbert-base-uncased"
ASPECTS = ["food_taste", "service", "price", "portion_size", "authenticity", "ambiance"]
LABELS = ["not_mentioned", "negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

MAX_LENGTH = 256
BATCH_SIZE = 8          # small batches - gold_train is only ~200 examples
LR = 1e-5                # lower LR than Phase 7 - fine-tuning further, not training fresh
MAX_EPOCHS = 15
PATIENCE = 3             # stop early if val macro-F1 doesn't improve for this many epochs


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
    def __init__(self, model_name, num_labels, aspects):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.2)
        self.heads = nn.ModuleDict({
            aspect: nn.Linear(hidden_size, num_labels) for aspect in aspects
        })

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        return {aspect: head(pooled) for aspect, head in self.heads.items()}


def evaluate(model, loader, device):
    model.eval()
    all_preds = {aspect: [] for aspect in ASPECTS}
    all_true = {aspect: [] for aspect in ASPECTS}
    with torch.no_grad():
        for batch in loader:
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
    mean_acc = sum(r["accuracy"] for r in results.values()) / len(results)
    mean_f1 = sum(r["macro_f1"] for r in results.values()) / len(results)
    return results, mean_acc, mean_f1, all_true, all_preds


def compute_class_weights(labels_by_id, review_ids, aspect):
    from collections import Counter
    counts = Counter(labels_by_id[rid][aspect] for rid in review_ids)
    total = sum(counts.values())
    weights = torch.zeros(len(LABELS))
    for label, idx in LABEL2ID.items():
        count = counts.get(label, 1)
        weights[idx] = total / (len(LABELS) * count)
    return weights


def main():
    device = get_device()
    print(f"Using device: {device}")

    with open("../data/processed/data_processed.json", encoding="utf-8") as f:
        reviews = json.load(f)
    text_by_id = {r["review_id"]: r["text"] for r in reviews}

    gold = load_jsonl("../data/processed/gold_labels.jsonl")
    gold_ids = [rid for rid in gold.keys() if rid in text_by_id]
    random.shuffle(gold_ids)

    n = len(gold_ids)
    n_train = int(n * 0.65)
    n_val = int(n * 0.15)
    train_ids = gold_ids[:n_train]
    val_ids = gold_ids[n_train:n_train + n_val]
    test_ids = gold_ids[n_train + n_val:]

    print(f"Gold split -> train: {len(train_ids)}  val: {len(val_ids)}  test: {len(test_ids)}")
    print("(test set here is smaller than the 319 used in Phase 6/7 - expect noisier numbers)")
    print()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = AspectDataset(train_ids, text_by_id, gold, tokenizer)
    val_ds = AspectDataset(val_ids, text_by_id, gold, tokenizer)
    test_ds = AspectDataset(test_ids, text_by_id, gold, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = MultiAspectDistilBert(MODEL_NAME, len(LABELS), ASPECTS).to(device)
    model.load_state_dict(torch.load("../model_multihead/model.pt", map_location=device))
    print("Loaded weak-trained weights from model_multihead/model.pt")

    # --- BEFORE: evaluate the weak-trained model on this held-out test slice ---
    print("\nEvaluating BEFORE gold fine-tuning (on held-out gold test slice)...")
    before_results, before_acc, before_f1, _, _ = evaluate(model, test_loader, device)
    print(f"Before: mean accuracy={before_acc*100:.1f}%  mean macro F1={before_f1:.3f}")

    # --- Fine-tune on gold_train, monitor on gold_val, early stopping on val macro F1 ---
    class_weights = {
        aspect: compute_class_weights(gold, train_ids, aspect).to(device) for aspect in ASPECTS
    }
    loss_fns = {aspect: nn.CrossEntropyLoss(weight=class_weights[aspect]) for aspect in ASPECTS}
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val_f1 = -1
    best_state = None
    epochs_without_improvement = 0

    print("\nFine-tuning on gold_train...")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
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
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        _, val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}  val_acc={val_acc*100:.1f}%  val_macro_f1={val_f1:.3f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"No improvement for {PATIENCE} epochs, stopping early.")
                break

    # --- AFTER: load best checkpoint (by val performance), evaluate on the held-out test slice ---
    model.load_state_dict(best_state)
    print("\nEvaluating AFTER gold fine-tuning (best checkpoint, on the SAME held-out test slice)...")
    after_results, after_acc, after_f1, all_true, all_preds = evaluate(model, test_loader, device)
    print(f"After: mean accuracy={after_acc*100:.1f}%  mean macro F1={after_f1:.3f}")

    print("\n=== Before vs After (same test slice, n={}) ===".format(len(test_ids)))
    print(f"{'Aspect':15s} {'Before Acc':>12s} {'After Acc':>12s} {'Before F1':>12s} {'After F1':>12s}")
    for aspect in ASPECTS:
        b = before_results[aspect]
        a = after_results[aspect]
        print(f"{aspect:15s} {b['accuracy']*100:11.1f}% {a['accuracy']*100:11.1f}% "
              f"{b['macro_f1']:12.3f} {a['macro_f1']:12.3f}")
    print(f"{'MEAN':15s} {before_acc*100:11.1f}% {after_acc*100:11.1f}% {before_f1:12.3f} {after_f1:12.3f}")

    for aspect in ASPECTS:
        y_true = [ID2LABEL[i] for i in all_true[aspect]]
        y_pred = [ID2LABEL[i] for i in all_preds[aspect]]
        print(f"\n=== {aspect} (after gold fine-tuning) ===")
        print(classification_report(y_true, y_pred, zero_division=0))

    os.makedirs("../results", exist_ok=True)
    with open("../results/gold_finetune_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "test_set_size": len(test_ids),
            "before": {"per_aspect": before_results, "mean_accuracy": before_acc, "mean_macro_f1": before_f1},
            "after": {"per_aspect": after_results, "mean_accuracy": after_acc, "mean_macro_f1": after_f1},
        }, f, indent=2)
    print("\nSaved results/gold_finetune_results.json")

    os.makedirs("../model_multihead_goldtuned", exist_ok=True)
    torch.save(best_state, "../model_multihead_goldtuned/model.pt")
    tokenizer.save_pretrained("../model_multihead_goldtuned")
    model.encoder.config.save_pretrained("../model_multihead_goldtuned")
    print("Saved gold-tuned model to model_multihead_goldtuned/")


if __name__ == "__main__":
    main()