"""
Local web app for manually labeling the gold evaluation sample.
Runs entirely on your machine - nothing sent anywhere.

Setup:
    pip install flask
    python labeling_app.py

Then open http://localhost:5050 in your browser.

Resumable: closes/reopens fine, skips reviews you've already labeled.

Input:  gold_sample.json   (the stratified sample to label)
Output: gold_labels.jsonl  (your labels, one line per review, appended as you go)
"""
import json
import os
from flask import Flask, request, redirect, url_for, render_template_string

SAMPLE_FILE = "../data/processed/gold_sample.json"
OUTPUT_FILE = "../data/processed/gold_labels.jsonl"

ASPECTS = [
    ("food_taste", "Food / Taste"),
    ("service", "Service"),
    ("price", "Price / Value"),
    ("portion_size", "Portion Size"),
    ("authenticity", "Authenticity"),
    ("ambiance", "Ambiance"),
]
SENTIMENTS = ["not_mentioned", "positive", "neutral", "negative"]

app = Flask(__name__)

with open(SAMPLE_FILE, encoding="utf-8") as f:
    SAMPLE = json.load(f)


def load_labeled_ids():
    done = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["review_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


PAGE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Gold Labeling</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; background: #fafafa; color: #222; }
  .progress { color: #666; margin-bottom: 20px; font-size: 14px; }
  .progress-bar { background: #e0e0e0; border-radius: 6px; height: 8px; overflow: hidden; margin-bottom: 24px; }
  .progress-fill { background: #2e7d32; height: 100%; transition: width 0.2s; }
  .meta { color: #888; font-size: 13px; margin-bottom: 8px; }
  .review-text { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 20px; font-size: 16px; line-height: 1.5; margin-bottom: 28px; }
  .stars { color: #f5a623; font-weight: bold; }
  .aspect-row { margin-bottom: 18px; }
  .aspect-label { font-weight: 600; margin-bottom: 6px; display: block; }
  .btn-group { display: flex; gap: 8px; }
  .btn-group input[type=radio] { display: none; }
  .btn-group label {
    flex: 1; text-align: center; padding: 10px 6px; border: 1px solid #ccc; border-radius: 6px;
    cursor: pointer; font-size: 13px; background: white; user-select: none;
  }
  .btn-group input[type=radio]:checked + label { color: white; border-color: transparent; }
  .opt-not_mentioned input:checked + label { background: #9e9e9e; color: white; }
  .opt-positive input:checked + label { background: #2e7d32; color: white; }
  .opt-neutral input:checked + label { background: #f9a825; color: white; }
  .opt-negative input:checked + label { background: #c62828; color: white; }
  .submit-btn {
    width: 100%; padding: 14px; font-size: 16px; background: #1565c0; color: white;
    border: none; border-radius: 8px; cursor: pointer; margin-top: 12px;
  }
  .submit-btn:hover { background: #0d47a1; }
  .done { text-align: center; padding: 60px 20px; }
</style>
</head>
<body>

{% if review %}
  <div class="progress">Labeled {{ done_count }} / {{ total }}</div>
  <div class="progress-bar"><div class="progress-fill" style="width: {{ pct }}%"></div></div>

  <div class="meta">{{ review.business_name }} &middot; {{ review.city }} &middot; {{ review.primary_cuisine }} &middot; <span class="stars">{{ '★' * review.stars|int }}</span></div>
  <div class="review-text">{{ review.text }}</div>

  <form method="POST" action="{{ url_for('submit') }}">
    <input type="hidden" name="review_id" value="{{ review.review_id }}">
    {% for key, display in aspects %}
      <div class="aspect-row">
        <span class="aspect-label">{{ display }}</span>
        <div class="btn-group">
          {% for s in sentiments %}
            <span class="opt-{{ s }}">
              <input type="radio" name="{{ key }}" id="{{ key }}_{{ s }}" value="{{ s }}" {% if s == 'not_mentioned' %}checked{% endif %}>
              <label for="{{ key }}_{{ s }}">{{ s.replace('_', ' ') }}</label>
            </span>
          {% endfor %}
        </div>
      </div>
    {% endfor %}
    <button type="submit" class="submit-btn">Save &amp; Next →</button>
  </form>
{% else %}
  <div class="done">
    <h2>All done!</h2>
    <p>You've labeled all {{ total }} reviews in the gold sample.</p>
    <p>Upload <code>gold_labels.jsonl</code> back to the chat.</p>
  </div>
{% endif %}

</body>
</html>
"""


@app.route("/")
def index():
    done_ids = load_labeled_ids()
    remaining = [r for r in SAMPLE if r["review_id"] not in done_ids]
    review = remaining[0] if remaining else None
    done_count = len(SAMPLE) - len(remaining)
    pct = round(done_count / len(SAMPLE) * 100) if SAMPLE else 0
    return render_template_string(
        PAGE_TEMPLATE, review=review, aspects=ASPECTS, sentiments=SENTIMENTS,
        done_count=done_count, total=len(SAMPLE), pct=pct,
    )


@app.route("/submit", methods=["POST"])
def submit():
    review_id = request.form["review_id"]
    label = {"review_id": review_id}
    for key, _ in ASPECTS:
        label[key] = request.form.get(key, "not_mentioned")
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(label, ensure_ascii=False) + "\n")
    return redirect(url_for("index"))


if __name__ == "__main__":
    print(f"Loaded {len(SAMPLE)} reviews to label.")
    print("Open http://localhost:5050 in your browser.")
    app.run(port=5050, debug=False)
