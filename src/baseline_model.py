"""
Phase 6: Baseline model.

Trains a simple, fast, interpretable baseline (TF-IDF + Logistic Regression)
per aspect, using the LLM weak labels as training data. Evaluates against
the human-labeled gold set - which is held out of training entirely, so
there's no leakage between "what the model learned from" and "what we
measure it against".

This baseline exists to answer one question: is a fine-tuned transformer
(Phase 7) actually worth the extra complexity? If the baseline already
gets 90%+ on some aspect, that's a real finding, not just a formality.
"""
import json
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

ASPECTS = ["food_taste", "service", "price", "portion_size", "authenticity", "ambiance"]


def load_jsonl(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                out[obj["review_id"]] = obj
    return out


def main():
    with open("../data/processed/data_processed.json", encoding="utf-8") as f:
        reviews = json.load(f)
    text_by_id = {r["review_id"]: r["text"] for r in reviews}

    weak = load_jsonl("../data/processed/labels_weak.jsonl")
    gold = load_jsonl("../data/processed/gold_labels.jsonl")

    # Gold review_ids are held out of training entirely - no leakage.
    gold_ids = set(gold.keys())
    train_ids = [rid for rid in weak.keys() if rid not in gold_ids and rid in text_by_id]
    test_ids = [rid for rid in gold.keys() if rid in text_by_id]

    print(f"Training examples: {len(train_ids)} (weak labels, gold reviews excluded)")
    print(f"Test examples: {len(test_ids)} (human gold labels, never seen in training)")
    print()

    results = {}
    for aspect in ASPECTS:
        X_train_text = [text_by_id[rid] for rid in train_ids]
        y_train = [weak[rid][aspect] for rid in train_ids]
        X_test_text = [text_by_id[rid] for rid in test_ids]
        y_test = [gold[rid][aspect] for rid in test_ids]

        vectorizer = TfidfVectorizer(
            max_features=20000, ngram_range=(1, 2), min_df=2, stop_words="english"
        )
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)

        clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        results[aspect] = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "train_label_dist": dict(Counter(y_train)),
            "test_label_dist": dict(Counter(y_test)),
        }

        print(f"=== {aspect} ===")
        print(f"  Accuracy vs gold: {acc*100:.1f}%")
        print(f"  Macro F1: {macro_f1:.3f}")
        print(classification_report(y_test, y_pred, zero_division=0))
        print()

    overall_acc = sum(r["accuracy"] for r in results.values()) / len(results)
    overall_f1 = sum(r["macro_f1"] for r in results.values()) / len(results)
    print(f"=== Overall (mean across aspects) ===")
    print(f"Mean accuracy: {overall_acc*100:.1f}%")
    print(f"Mean macro F1: {overall_f1:.3f}")

    with open("baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {"per_aspect": results, "overall_accuracy": overall_acc, "overall_macro_f1": overall_f1},
            f, indent=2,
        )
    print()
    print("Saved baseline_results.json")


if __name__ == "__main__":
    main()
