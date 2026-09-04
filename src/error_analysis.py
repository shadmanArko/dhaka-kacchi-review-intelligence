"""
Phase 8: Error analysis.

Runs the gold-tuned model (Phase 7.5) against the val+test portions of
the gold set (the 112 reviews NOT used to update model weights - val was
used only for early-stopping selection, test was never touched at all).
Training-portion reviews are excluded here on purpose: the model has
directly learned from them, so errors there wouldn't reflect real
generalization failures.

For each misclassified example, prints the full review text alongside
the true vs predicted label, so you can read them and look for patterns
(sarcasm, mixed sentiment, short reviews, implied-not-stated aspects,
etc.). Also breaks errors down by cuisine and star rating to check for
systematic bias.

Setup: same dependencies as the other Phase 7 scripts.
    python error_analysis.py

Output: results/error_analysis.json (all errors, structured)
        prints a readable report to the console
"""
import json
import random
from collections import Counter, defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

MODEL_NAME = "distilbert-base-uncased"
ASPECTS = ["food_taste", "service", "price", "portion_size", "authenticity", "ambiance"]
LABELS = ["not_mentioned", "negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
MAX_LENGTH = 256


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
            "review_id": rid,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        for aspect in ASPECTS:
            item[f"label_{aspect}"] = torch.tensor(
                LABEL2ID[self.labels_by_id[rid][aspect]], dtype=torch.long
            )
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


def main():
    device = get_device()
    print(f"Using device: {device}")

    with open("../data/processed/data_processed.json", encoding="utf-8") as f:
        reviews = json.load(f)
    text_by_id = {r["review_id"]: r["text"] for r in reviews}
    meta_by_id = {r["review_id"]: r for r in reviews}

    gold = load_jsonl("../data/processed/gold_labels.jsonl")
    gold_ids = [rid for rid in gold.keys() if rid in text_by_id]

    # Reproduce the EXACT same split gold_finetune.py used, so we know which
    # ids were train (seen by the model) vs val/test (not used for weight updates)
    shuffled = gold_ids.copy()
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 0.65)
    n_val = int(n * 0.15)
    train_ids = set(shuffled[:n_train])
    eval_ids = shuffled[n_train:]  # val + test combined - neither updated model weights

    print(f"Analyzing errors on {len(eval_ids)} reviews (val+test - excludes {len(train_ids)} training reviews)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = AspectDataset(eval_ids, text_by_id, gold, tokenizer)
    loader = DataLoader(ds, batch_size=8)

    model = MultiAspectDistilBert(MODEL_NAME, len(LABELS), ASPECTS).to(device)
    model.load_state_dict(torch.load("../model_multihead_goldtuned/model.pt", map_location=device))
    model.eval()

    errors = []  # one entry per (review, aspect) mismatch
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            for aspect in ASPECTS:
                preds = logits[aspect].argmax(dim=-1).cpu().tolist()
                trues = batch[f"label_{aspect}"].tolist()
                for i, rid in enumerate(batch["review_id"]):
                    if preds[i] != trues[i]:
                        errors.append({
                            "review_id": rid,
                            "aspect": aspect,
                            "true": ID2LABEL[trues[i]],
                            "predicted": ID2LABEL[preds[i]],
                            "text": text_by_id[rid],
                            "cuisine": meta_by_id[rid]["primary_cuisine"],
                            "stars": meta_by_id[rid]["stars"],
                            "word_count": len(text_by_id[rid].split()),
                        })

    print(f"\nTotal errors: {len(errors)} across {len(eval_ids)} reviews x {len(ASPECTS)} aspects "
          f"({len(errors)/(len(eval_ids)*len(ASPECTS))*100:.1f}% of all judgments)")

    # --- Breakdown by aspect ---
    print("\n=== Errors by aspect ===")
    aspect_counts = Counter(e["aspect"] for e in errors)
    for aspect, count in aspect_counts.most_common():
        print(f"  {aspect:15s}: {count}")

    # --- Breakdown by cuisine (normalized by how many reviews of that cuisine exist) ---
    print("\n=== Errors by cuisine (raw counts) ===")
    cuisine_counts = Counter(e["cuisine"] for e in errors)
    for cuisine, count in cuisine_counts.most_common():
        print(f"  {cuisine:15s}: {count}")

    # --- Breakdown by review length ---
    print("\n=== Error review length stats ===")
    lengths = [e["word_count"] for e in errors]
    all_lengths = [len(text_by_id[rid].split()) for rid in eval_ids]
    print(f"  Mean word count of MISCLASSIFIED reviews: {sum(lengths)/len(lengths):.0f}")
    print(f"  Mean word count of ALL reviews in eval set: {sum(all_lengths)/len(all_lengths):.0f}")

    # --- Most common confusion patterns (true -> predicted) ---
    print("\n=== Most common confusion patterns (true -> predicted) ===")
    confusions = Counter((e["true"], e["predicted"]) for e in errors)
    for (true, pred), count in confusions.most_common(10):
        print(f"  {true:15s} -> {pred:15s}: {count}")

    # --- Sample misclassified reviews for qualitative reading ---
    print("\n=== Sample misclassified reviews (up to 10) ===")
    random.shuffle(errors)
    for e in errors[:10]:
        print(f"\n[{e['aspect']}] true={e['true']} predicted={e['predicted']} "
              f"cuisine={e['cuisine']} stars={e['stars']}")
        print(f"  \"{e['text'][:300]}{'...' if len(e['text']) > 300 else ''}\"")

    import os
    os.makedirs("results", exist_ok=True)
    with open("../results/error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(errors)} error examples to results/error_analysis.json")


if __name__ == "__main__":
    main()
