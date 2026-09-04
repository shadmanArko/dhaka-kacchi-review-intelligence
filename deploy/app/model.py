"""
Model definition and loading for deployment. Architecture must exactly
match the training scripts (src/finetune_multihead.py, src/gold_finetune.py)
since we're loading weights trained there.
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig

MODEL_NAME = "distilbert-base-uncased"
ASPECTS = ["food_taste", "service", "price", "portion_size", "authenticity", "ambiance"]
LABELS = ["not_mentioned", "negative", "neutral", "positive"]
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
MAX_LENGTH = 256

MODEL_DIR = "model_weights"  # weights + tokenizer + config copied into the image at this path
CONFIDENCE_THRESHOLD = 0.65  # below this, report "uncertain" rather than asserting a label -
                              # Phase 8 error analysis showed misclassifications cluster right
                              # around 50-55% confidence, so this threshold catches most of them


class MultiAspectDistilBert(nn.Module):
    def __init__(self, config, num_labels, aspects):
        super().__init__()
        # Built from config only, NOT from_pretrained - no need to download the
        # base pretrained weights since our own fine-tuned state_dict overwrites
        # them immediately below. Faster cold start, no runtime network dependency.
        self.encoder = AutoModel.from_config(config)
        hidden_size = config.hidden_size
        self.dropout = nn.Dropout(0.2)
        self.heads = nn.ModuleDict({
            aspect: nn.Linear(hidden_size, num_labels) for aspect in aspects
        })

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        return {aspect: head(pooled) for aspect, head in self.heads.items()}


class ReviewAnalyzer:
    """Loads the model once and exposes a simple .predict(text) method."""

    def __init__(self, model_dir=MODEL_DIR):
        self.device = torch.device("cpu")  # CPU inference - fine for a single-review API,
                                            # and avoids needing a GPU instance in AWS
        # Both tokenizer and config are loaded from the bundled local files
        # (copied into the image at build time) - no Hugging Face Hub calls
        # at runtime, so the container works even with restricted network access.
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        config = AutoConfig.from_pretrained(model_dir)
        self.model = MultiAspectDistilBert(config, len(LABELS), ASPECTS)
        state_dict = torch.load(f"{model_dir}/model.pt", map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        enc = self.tokenizer(
            text, truncation=True, max_length=MAX_LENGTH,
            padding="max_length", return_tensors="pt",
        )
        logits = self.model(enc["input_ids"], enc["attention_mask"])
        result = {}
        for aspect in ASPECTS:
            probs = torch.softmax(logits[aspect], dim=-1)[0]
            pred_id = probs.argmax().item()
            confidence = round(probs[pred_id].item(), 3)
            sentiment = ID2LABEL[pred_id]
            # Below the confidence threshold, report honest uncertainty instead of
            # asserting a specific label the model isn't actually sure about -
            # "not_mentioned" is exempt since a low-confidence "not mentioned" is
            # still a reasonably safe default, unlike asserting a specific sentiment
            if confidence < CONFIDENCE_THRESHOLD and sentiment != "not_mentioned":
                sentiment = "uncertain"
            result[aspect] = {
                "sentiment": sentiment,
                "confidence": confidence,
            }
        return result