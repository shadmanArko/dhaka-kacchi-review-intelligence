---
license: apache-2.0
base_model: distilbert-base-uncased
tags:
  - absa
  - aspect-based-sentiment-analysis
  - sentiment-analysis
  - restaurant-reviews
  - distilbert
language: en
pipeline_tag: text-classification
---

# Dhaka Kacchi Review Intelligence — Aspect-Based Sentiment Model

A multi-head DistilBERT model that predicts sentiment for **six independent aspects** of a
restaurant review, rather than collapsing it into a single score.

Full project write-up, data pipeline, and evaluation: [GitHub — dhaka-kacchi-review-intelligence](https://github.com/shadmanArko/dhaka-kacchi-review-intelligence).
Live demo: https://reviews.dhakakacchi.com

## Model description

- **Base:** `distilbert-base-uncased`, with six linear classification heads sharing one encoder.
- **Aspects:** `food_taste`, `service`, `price`, `portion_size`, `authenticity`, `ambiance`.
- **Classes per aspect:** `positive`, `negative`, `neutral`, `not_mentioned`.
- **Training:** weak-label pretraining (~18.5k LLM-labeled Yelp reviews) followed by fine-tuning
  on 207 human-labeled gold reviews.

## Evaluation

Measured on 65 gold reviews held out entirely from training (never seen in weak-label
pretraining or gold fine-tuning):

| Aspect | Accuracy | Macro F1 |
|---|---|---|
| `food_taste` | 81.5% | 0.590 |
| `service` | 80.0% | 0.573 |
| `price` | 89.2% | 0.706 |
| `portion_size` | 92.3% | 0.591 |
| `authenticity` | 95.4% | 0.640 |
| `ambiance` | 84.6% | 0.671 |
| **Mean** | **87.2%** | **0.628** |

See the [error analysis](https://github.com/shadmanArko/dhaka-kacchi-review-intelligence#10-error-analysis)
in the project README for known failure modes (over-triggering on `not_mentioned`, a 256-token
truncation limit, and a single-annotator gold set).

## Usage

```python
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel
from huggingface_hub import hf_hub_download

REPO = "shadmanArko/dhaka-kacchi-review-intelligence"
ASPECTS = ["food_taste", "service", "price", "portion_size", "authenticity", "ambiance"]
LABELS = ["positive", "negative", "neutral", "not_mentioned"]

tokenizer = AutoTokenizer.from_pretrained(REPO)
config = AutoConfig.from_pretrained(REPO)

class MultiAspectDistilBert(torch.nn.Module):
    def __init__(self, config, n_labels, aspects):
        super().__init__()
        self.encoder = AutoModel.from_config(config)
        self.heads = torch.nn.ModuleDict({
            a: torch.nn.Linear(config.dim, n_labels) for a in aspects
        })

    def forward(self, input_ids, attention_mask):
        pooled = self.encoder(input_ids, attention_mask).last_hidden_state[:, 0]
        return {a: head(pooled) for a, head in self.heads.items()}

model = MultiAspectDistilBert(config, len(LABELS), ASPECTS)
state_dict = torch.load(hf_hub_download(REPO, "model.pt"), map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

text = "The biryani tastes good but the price is very high. Service was not good."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
with torch.no_grad():
    logits = model(inputs["input_ids"], inputs["attention_mask"])

for aspect, out in logits.items():
    pred = LABELS[out.argmax(-1).item()]
    print(f"{aspect}: {pred}")
```

## Limitations

- 256-token truncation drops content in ~11% of reviews.
- Trained predominantly on US/Canadian Yelp reviews; underrepresented vocabulary for
  Bangladeshi-cuisine-specific terms (e.g. "biryani").
- Single-annotator gold set; `neutral` remains the weakest class across all aspects.

See the [project README](https://github.com/shadmanArko/dhaka-kacchi-review-intelligence) for
the full list of known limitations and next steps.
