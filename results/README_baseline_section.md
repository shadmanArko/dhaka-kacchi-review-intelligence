## Baseline Model (Phase 6)

Before fine-tuning a transformer, a TF-IDF + Logistic Regression baseline was trained per
aspect to establish a defensible performance floor and justify the added complexity of a
neural model.

**Setup:** one binary-per-class Logistic Regression classifier per aspect (`food_taste`,
`service`, `price`, `portion_size`, `authenticity`, `ambiance`), trained on ~19.5k
LLM-weak-labeled reviews, evaluated against a 319-review human-labeled gold set that was
held out of training entirely.

![Baseline accuracy vs macro F1](results/baseline_accuracy_vs_f1.png)

| Aspect | Accuracy | Macro F1 |
|---|---|---|
| food_taste | 65.8% | 0.462 |
| service | 69.0% | 0.476 |
| price | 80.9% | 0.497 |
| portion_size | 81.8% | 0.443 |
| authenticity | 89.3% | 0.492 |
| ambiance | 75.9% | 0.513 |
| **Mean** | **77.1%** | **0.481** |

**Key finding:** accuracy is inflated by class imbalance — every aspect is dominated by
`not_mentioned` (e.g. 288/319 for authenticity), so a model can score well just by
defaulting to the majority class. Macro F1 exposes this: the `neutral` class scores
0.00-0.18 F1 across every aspect, since it makes up under 7% of training examples per
aspect. This motivates fine-tuning a pretrained transformer (Phase 7), which brings general
language understanding that doesn't depend on large per-class example counts to learn a
concept like "mixed sentiment."

Full per-class precision/recall and training code: [`src/baseline_model.py`](src/baseline_model.py),
raw results: [`results/baseline_results.json`](results/baseline_results.json).
