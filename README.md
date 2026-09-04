# Dhaka Kacchi Review Intelligence

An aspect-based sentiment analysis (ABSA) system for restaurant reviews: given a review, it
extracts sentiment toward six distinct aspects independently, rather than reducing the review
to a single overall score.

**Live demo:** https://reviews.dhakakacchi.com

Built around [Dhaka Kacchi](https://dhakakacchi.com), a Berlin cloud kitchen, as an end-to-end
ML engineering exercise: real messy data, weak supervision at scale, a baseline-versus-neural
comparison, honest error analysis, and a containerized cloud deployment.

---

## Table of contents

- [Problem framing](#1-problem-framing)
- [Data sourcing — including two dead ends](#2-data-sourcing--including-two-dead-ends)
- [Data integrity](#3-data-integrity)
- [Exploratory analysis](#4-exploratory-analysis)
- [Preprocessing](#5-preprocessing)
- [Labeling strategy](#6-labeling-strategy-weak-supervision--a-human-gold-set)
- [Baseline model](#7-baseline-model)
- [Fine-tuned transformer](#8-fine-tuned-transformer)
- [Breaking the label-noise ceiling](#9-breaking-the-label-noise-ceiling)
- [Error analysis](#10-error-analysis)
- [Deployment](#11-deployment)
- [Engineering practices applied](#engineering-practices-applied)
- [Known limitations](#known-limitations)
- [Reproducing this project](#reproducing-this-project)

---

## 1. Problem framing

**Business question:** which aspects of the dining experience drive negative sentiment, and how
does that change over time — without manually reading hundreds of reviews?

A single star rating collapses everything into one number. A 3-star review saying *"amazing food,
terrible service"* is not the same as one saying *"mediocre everything"*, but both average out
identically. Aspect-based sentiment analysis keeps them distinct.

**Target aspects** (six, deliberately scoped):

| Aspect | Definition |
|---|---|
| `food_taste` | Quality and taste of the food itself |
| `service` | Staff friendliness, speed, attentiveness |
| `price` | Value for money; felt expensive or cheap |
| `portion_size` | Generous or small portions |
| `authenticity` | Felt authentic/traditional vs. inauthentic |
| `ambiance` | Atmosphere, decor, noise, cleanliness of the space |

**Output classes per aspect:** `positive`, `negative`, `neutral`, `not_mentioned`.

**Scoping decisions made here, and why:**

- *Delivery and wait time were dropped.* Initially included, but the cloud-kitchen context made
  them less actionable than the rest, and every extra aspect multiplies labeling cost.
- *`portion_size` and `authenticity` were kept.* These matter disproportionately for
  rice-and-meat-forward cuisines, where generous portions and "tastes like home" are core to how
  customers actually judge a meal. They are also absent from most off-the-shelf ABSA datasets,
  which is part of why this project labels its own data.
- *`not_mentioned` is a first-class class, not a null.* Most reviews discuss two or three aspects,
  not all six. Treating absence as a real prediction target is what makes the output honest —
  and, as it turns out, it is also where the model struggles most (see
  [error analysis](#10-error-analysis)).

---

## 2. Data sourcing — including two dead ends

Dhaka Kacchi does not yet have enough of its own review volume to train on, so the pipeline was
built and validated on public restaurant-review data, with the intent of applying it to real
business data later.

Two candidate datasets were rejected after inspection. Both are documented here because
**vetting data before building on it is the work**, and the failure modes are instructive.

### Rejected: a "Yelp Restaurant Reviews" CSV (20k rows)

Schema looked fine — review text, rating, date, business URL. No missing values. Reasonable
length distribution.

Then extracting business names from the URLs revealed the problem: **all 49 businesses were
dessert and bakery shops** — ice cream parlours, donut shops, crêperies, cupcake bakeries. Zero
full-service restaurants, zero South Asian cuisine.

That is unusable here, and not for cosmetic reasons: **`portion_size` and `authenticity` are
close to meaningless for an ice cream scoop.** Nobody reviews gelato for authenticity the way
they review biryani. Forcing it would have produced a model that could not demonstrate the
aspects the project exists to measure.

### Rejected: a "Restaurant Reviews Data" CSV with a `category` column

This looked ideal — it advertised a cuisine `category` field, which the first dataset lacked.

Inspecting actual rows: `review_text` values were random alphanumeric strings
(`JGYylwLXKQ`, `aAcFSNyaOO`). Cities were generated placeholders (`Fort Bernie`, `Heaneystad`,
`Madelineburgh`). And `category` turned out to be *meal type* (Breakfast/Lunch/Dinner), not
cuisine at all.

This was **synthetic mock data**, almost certainly Faker-generated to test a database schema.
There is no language in it for an NLP model to learn from. A pipeline built on it would have run
end-to-end and produced meaningless metrics.

### Also evaluated: GERestaurant (German, pre-annotated)

A manually annotated German TripAdvisor ABSA dataset (~3k reviews) from a 2024 University of
Regensburg paper. Genuinely attractive — German-language data is *directly* relevant to a Berlin
business, and expert annotations would have removed the need to build a labeling pipeline.

Rejected on availability: the dataset is gated ("available upon request from the authors"), and
the public GitHub repo contains only the code and restaurant list, not the annotated reviews.
Noted as a strong candidate for a future German-language iteration.

### Selected: the official Yelp Open Dataset

Downloaded directly from Yelp rather than a third-party re-upload — real data, genuine
`categories` fields, no dependency on someone else's cleaning or labeling.

**Filtering approach:** rather than trusting a pre-filtered subset, the raw 150,346-business
file was filtered locally to cuisines matching the target profile — rice-and-meat-forward,
generous portions, judged partly on authenticity:

> Indian, Pakistani, Bangladeshi, Middle Eastern, Lebanese, Turkish, Persian, Halal, Afghan,
> Mediterranean, Caribbean, Thai, Vietnamese, Malaysian, Indonesian, Nepalese, Sri Lankan,
> Moroccan, Ethiopian, Somali

This is deliberately **international, not South Asian only** — the target market is Berlin's
diaspora *and* the broader set of people who eat this style of food, so cuisine breadth is a
feature, not noise.

Result: **4,867 matching restaurants, ~476,000 available reviews** across dozens of US and
Canadian cities. Bangladeshi cuisine specifically had only 1 matching business on Yelp — a real
gap, and a documented source of domain mismatch for the eventual Dhaka Kacchi use case.

**Review extraction:** the full review file is multi-gigabyte, so a streaming filter script
([`src/filter_reviews.py`](src/filter_reviews.py)) reads it line-by-line, keeps only reviews for
matched businesses, and caps at **150 reviews per business** so no single high-volume restaurant
dominates the corpus — a direct lesson from the first rejected dataset, where 49 businesses
accounted for everything.

**A limitation of that script, stated plainly:** it stops at a 20,000-review cap in file order,
so it collected 20,000 reviews across **428 businesses** rather than sampling evenly across all
4,867. The result is still spread across 12 cuisines and 10+ cities, but it is a convenience
sample, not a uniform one.

---

## 3. Data integrity

Final working set: **20,000 reviews → 19,837 after cleaning**, 428 restaurants, 14 cuisines.

Schema carried through the pipeline:

```
review_id, business_id, business_name, city, state, primary_cuisine, stars, date, text
```

Verified before proceeding:

- No missing values in any column
- No duplicate `review_id`s
- **26 exact-duplicate review texts** (same text, different IDs — a Yelp data artifact) → dropped
- Realistic rating distribution: 46% 5-star, 27% 4-star, tapering down — matching the known
  positive skew of real review platforms rather than a suspiciously uniform spread

### A labeling ambiguity worth catching

Yelp's `Indian` category tag is **ambiguous between South Asian and Native American cuisine.**
The filter matched *"Indian Frybread — Manna From Heaven"*, whose full categories read
`Restaurants, Mexican, American (Traditional), Indian, Food` — frybread, not biryani.

A second false positive, *"India House"*, was tagged `Home & Garden, Home Decor, Shopping` —
an import store carrying a stray `Restaurants` tag.

Both were dropped (137 reviews). Checking all ~76 Indian-tagged businesses confirmed these were
isolated, not a systemic failure of the filter. **This class of error does not show up in any
summary statistic** — it required reading the business names.

---

## 4. Exploratory analysis

- **Date range:** 2005–2020, concentrated 2011–2018 (peak 2017, n=3,399). The 2019–2020 tail is
  sparse — relevant to any time-series framing, not a problem for static classification.
- **Review length:** median 69 words, mean 97, max 946 — right-skewed, typical of review text.
- **Very short reviews:** only 71 under 10 words. **Kept deliberately** — *"Food was cold"* is a
  dense, valid aspect statement, not noise. Dropping short reviews would have removed exactly the
  unambiguous examples a model most needs.
- **Rating by cuisine:** Thai, Indian, Mediterranean, Caribbean, and Middle Eastern cluster
  tightly at 3.9–4.1. Moroccan stood out at **2.93 (n=101)** — flagged as either genuine signal
  or a single low-rated business skewing a small sample; not resolved, and noted rather than
  quietly ignored.

---

## 5. Preprocessing

`src/preprocess.py` *(not preserved in version control; its committed output lives in
`data/processed/`)* — deterministic, and the output is committed so the
exact training input is inspectable.

1. Drop 26 exact-duplicate texts
2. Drop the 2 mislabeled businesses (137 reviews)
3. Parse `categories` into a single `primary_cuisine` field
4. Normalize whitespace — **punctuation deliberately preserved**, since it carries sentiment
   signal (`"Amazing!!!"` ≠ `"Amazing"`)

**One ordering decision worth explaining:** South Asian restaurants on Yelp frequently carry both
`Pakistani` and `Indian` tags. The cuisine-matching order checks specific tags *before* general
ones, so a restaurant tagged both resolves to `Pakistani` — the more specific signal — rather
than whichever appeared first alphabetically.

**Final cuisine distribution (19,837 reviews):**

| Cuisine | n | Cuisine | n |
|---|---|---|---|
| Thai | 5,104 | Turkish | 320 |
| Vietnamese | 3,237 | Halal | 152 |
| Mediterranean | 2,905 | Indonesian | 105 |
| Caribbean | 2,023 | Moroccan | 101 |
| Indian | 1,920 | Afghan | 63 |
| Middle Eastern | 1,795 | Persian | 46 |
| Pakistani | 1,652 | | |
| Ethiopian | 414 | | |

---

## 6. Labeling strategy: weak supervision + a human gold set

Yelp reviews carry star ratings, not aspect labels. Hand-labeling 19,837 reviews × 6 aspects
(≈119,000 judgments) was not feasible.

**The approach — standard current industry practice for exactly this situation:**

1. **LLM weak labeling** across the full corpus — fast, cheap, imperfect
2. **A human-labeled gold set** on a stratified sample — small, trustworthy
3. **Measure the weak labels against gold** — turning "we used an LLM" into a defensible number
4. Gold becomes the evaluation set; weak labels become training data

### Weak labeling: an infrastructure detour worth documenting

**Attempt 1 — Groq API, `llama-3.3-70b-versatile`:** returned `404 model_not_found`. The model
had been **deprecated on 2026-08-16**, weeks before this project. Switched to Groq's recommended
replacement, `openai/gpt-oss-120b`.

**Attempt 2 — same API, new model:** every batch failed to parse, returning empty strings.
Cause: `gpt-oss-120b` is a **reasoning model**, and it was exhausting its output token budget on
internal reasoning before emitting any JSON. Fixed with two changes — `response_format:
json_object` to enforce structured output at the API level rather than hoping the prompt held,
and `reasoning_effort: low`, since aspect labeling does not need deep deliberation.

**Attempt 3 — working, but the arithmetic did not:** labeling succeeded, then hit Groq's free-tier
cap of **200,000 tokens/day**. Measured cost was ~2,791 tokens per 10-review batch, giving a
sustainable throughput of roughly **717 reviews/day → ~28 days** for the full corpus.

The script was made resumable and rate-limit-aware regardless (parsing Groq's own
`"try again in Xm Ys"` hint and waiting exactly that long, rather than guessing or quitting) —
but four weeks for one pipeline stage was not an acceptable plan.

**Options considered:** wait it out; reduce scope to a smaller stratified sample; pay for Groq's
Developer tier; run inference locally. Notably **rejected: creating additional accounts to
multiply free-tier quota** — that violates Groq's terms of service, and "it would have worked"
is not a justification.

**Resolution — local inference via Ollama** (`gpt-oss:20b`, the same model family as the Groq
model, so the prompt transferred unchanged). No rate limits, no cost, no daily caps. Two further
problems surfaced:

- **Truncated JSON mid-string.** `finish_reason: length` confirmed output truncation. Raising
  `max_tokens` did not help, because Ollama's **default 4,096-token context window** was the
  actual ceiling — and that window is shared between prompt, reasoning, and response.
- **The context setting silently did not apply.** Requests through Ollama's OpenAI-compatibility
  layer were not forwarding `num_ctx` to the model. Verified by unloading the model and
  confirming an identical failure at an identical character offset. Fixed by switching to
  Ollama's **native `/api/chat` endpoint**, where `num_ctx` and `num_predict` are first-class
  parameters.

**Final labeling pipeline** ([`src/label_reviews_ollama.py`](src/label_reviews_ollama.py)):

- Local `gpt-oss:20b`, native Ollama API, 12,000-token context, batches of 6
- **Schema validation before writing** — malformed responses are skipped, not written; bad labels
  never silently enter the dataset
- **Recursive batch splitting** — a batch that fails 3 retries splits in half and retries each
  half, down to single reviews. One pathological review can no longer cost five good ones.
- **Resumable by design** — reads already-labeled IDs on startup and skips them, so the job
  survives interruption, sleep, and restarts with no duplicate work
- Throughput: ~0.8 batches/min ≈ 288 reviews/hour → **~2.8 days** unattended, versus 28

**Honest caveat:** ~780 reviews (≈4%) carry labels from the Groq run before the pivot; the
remaining ~96% are from the local model. Same model family and identical prompt, but not
identical weights — a small source of label heterogeneity.

### The gold evaluation set

`src/build_gold_sample.py` *(not preserved in version control; its committed output is
`data/processed/gold_sample.json`)* — **319 reviews**, stratified
proportionally across all 14 cuisines with a floor of 8 per cuisine so small cuisines (Persian,
n=46) are genuinely represented rather than rounded out of existence. Seeded (`random_state=42`)
and shuffled, so the sample is reproducible.

Labeled by hand through a purpose-built local Flask app
([`src/labeling_app.py`](src/labeling_app.py)) — one review at a time, six aspect rows, four
colour-coded buttons each, `not_mentioned` pre-selected as the safe default. Resumable, writing
one line per review as it goes. **1,914 individual human judgments.**

Building a tool for this rather than editing JSON by hand is not an aesthetic choice: at 6
judgments per review, friction is what determines whether a labeling task actually gets
finished.

### Measured weak-label quality

**79.0% exact-match agreement with human gold labels** (1,512 / 1,914) — well above the 25%
four-class random baseline.

| Aspect | Agreement |
|---|---|
| `authenticity` | 93.1% |
| `portion_size` | 85.6% |
| `price` | 81.2% |
| `ambiance` | 78.1% |
| `food_taste` | **68.3%** |
| `service` | **67.7%** |

Decomposing the two weak aspects into *detection* (is the aspect mentioned at all?) versus
*polarity* (given both agree it is mentioned, is the sentiment right?) localized the failure
precisely:

| Aspect | Detection | Polarity (both agree mentioned) |
|---|---|---|
| `food_taste` | 76.5% | **88.6%** |
| `service` | 72.7% | **86.6%** |

**The weak labeler is not confused about sentiment — it over-triggers.** It projects a review's
overall positive tone onto the most salient aspects, calling `food_taste` positive when the
review never actually describes the food. 47 of the `food_taste` errors were exactly this
pattern. Polarity accuracy stays high at 86–89%.

This bias is carried forward as a known property of the training data, and it reappears —
measurably — in every model trained on it.

---

## 7. Baseline model

[`src/baseline_model.py`](src/baseline_model.py) — TF-IDF (1–2 grams, 20k features) +
Logistic Regression with balanced class weights, one classifier per aspect.

**This baseline exists to answer a specific question:** is a fine-tuned transformer actually
worth the complexity? It is also, deliberately, a **no-transfer-learning control** — it learns
purely from this corpus, with no pretrained language knowledge, which makes it the honest
comparison point for measuring what pretraining is worth.

**Split discipline:** the 319 gold reviews are excluded from training entirely. Trained on
~19,500 weak labels, evaluated on human gold labels never seen in training.

| Aspect | Accuracy | Macro F1 |
|---|---|---|
| `food_taste` | 65.8% | 0.462 |
| `service` | 69.0% | 0.476 |
| `price` | 80.9% | 0.497 |
| `portion_size` | 81.8% | 0.443 |
| `authenticity` | 89.3% | 0.492 |
| `ambiance` | 75.9% | 0.513 |
| **Mean** | **77.1%** | **0.481** |

**The gap between those two columns is the finding.** Every aspect is dominated by
`not_mentioned` (288/319 for authenticity), so accuracy rewards a model that simply defaults to
the majority class. Macro F1 — weighting all four classes equally — exposes it.

**The `neutral` class is effectively unlearnable for this model:** 0.00–0.18 F1 across every
aspect, with precision and recall both 0.00 for `service`. With 6–21 `neutral` training examples
per aspect out of ~19,500, a linear bag-of-ngrams model has no path to learning what "mixed
sentiment" means.

That is a concrete, evidenced motivation for the next phase — not a formality.

---

## 8. Fine-tuned transformer

[`src/finetune_multihead.py`](src/finetune_multihead.py)

### Architecture: one shared encoder, six heads

**Base model:** `distilbert-base-uncased` — pretrained by Hugging Face on BookCorpus and English
Wikipedia via masked language modeling. **This is where transfer learning happens:** the
pretrained weights are loaded as the starting point (general English understanding — grammar,
word sense, how context shifts meaning), six randomly-initialized linear heads are attached, and
encoder and heads are updated together during training.

**The design choice — multi-task learning over six independent models:**

| | Six separate models | Shared encoder + 6 heads |
|---|---|---|
| Training cost | 6× full DistilBERT | ~1× |
| Deployment artifact | 6 models (~260MB each) | 1 model |
| Rare classes (`neutral`) | No cross-aspect transfer | Heads share the encoder's reading ability |

The reasoning: understanding *"tasted just like home"* as a cultural comparison, or connecting
*"tiny portions for the price"* as a single value judgment, is **general reading comprehension**.
Learning it once and sharing it across six heads means low-data aspects borrow strength from
high-data ones, instead of each head relearning English from scratch.

**Training:** 18,518 train / 1,000 weak-label validation / 319 gold test. 3 epochs, batch size
16, max length 256, AdamW at 2e-5 with linear warmup, gradient clipping at 1.0. **Per-aspect
class weights** computed from training distribution — a direct, targeted response to the
`neutral` collapse in the baseline. ~34 minutes on an M3 Pro via PyTorch MPS.

| Aspect | Accuracy | Macro F1 | Δ F1 vs baseline |
|---|---|---|---|
| `food_taste` | 64.6% | 0.452 | −0.010 |
| `service` | 67.7% | 0.504 | +0.028 |
| `price` | 79.3% | 0.518 | +0.021 |
| `portion_size` | 84.3% | 0.474 | +0.031 |
| `authenticity` | 88.7% | 0.455 | −0.037 |
| `ambiance` | 77.7% | 0.557 | +0.044 |
| **Mean** | **77.1%** | **0.493** | **+0.012** |

### The result: essentially a tie

Mean accuracy identical to the baseline (77.1%). Macro F1 up 2.5% relative. Two aspects got
*worse*. A 66M-parameter pretrained transformer performed level with TF-IDF and logistic
regression.

**This is the most important finding in the project, and it is not a failure to explain away.**

Both models trained on the same weak labels, measured at 79% accuracy with a documented
over-triggering bias. **A more capable model does not repair a noisy training signal — it learns
the same patterns more confidently, bias included.** DistilBERT faithfully reproduced the weak
labeler's mistakes because it was never shown anything better.

Stated as the conclusion it is: *model capacity was not the binding constraint. Label quality
was.*

---

## 9. Breaking the label-noise ceiling

[`src/gold_finetune.py`](src/gold_finetune.py)

If label noise is the ceiling, a small amount of *real* signal should break through it. That is a
falsifiable prediction, so it was tested — weak-supervision pretraining followed by a small
gold-label fine-tune.

**Split, and its cost:** fine-tuning on gold means gold can no longer be the full test set. The
319 reviews were split 207 train / 47 validation (early-stopping only) / **65 held out entirely**.
Learning rate lowered to 1e-5 (continuing training, not starting fresh), early stopping on
validation macro F1 with patience 3 — stopped at epoch 13, best checkpoint from epoch 10.

**Before and after, on the same never-trained-on 65 reviews:**

| Aspect | Acc before → after | Macro F1 before → after |
|---|---|---|
| `food_taste` | 76.9% → 81.5% | 0.547 → 0.590 |
| `service` | 80.0% → 80.0% | 0.538 → 0.573 |
| `price` | 84.6% → 89.2% | 0.643 → 0.706 |
| `portion_size` | 80.0% → 92.3% | 0.475 → 0.591 |
| `authenticity` | 87.7% → 95.4% | 0.558 → 0.640 |
| `ambiance` | 75.4% → 84.6% | 0.531 → 0.671 |
| **Mean** | **80.8% → 87.2%** | **0.549 → 0.628** |

**+6.4 accuracy points and +14% relative macro F1 — from 207 human-labeled examples**, against
the ~18,500 weak-labeled reviews already in the model. Every aspect improved. The hypothesis
held.

**Required caveat:** this "before" figure (80.8%) is measured on 65 reviews, not the 319 used in
sections 7–8. **It is not comparable to the 77.1% headline** — the only valid comparison is
before-versus-after on this same 65-review slice. Small test sets make per-aspect numbers noisy,
particularly for rare classes.

---

## 10. Error analysis

[`src/error_analysis.py`](src/error_analysis.py) — the gold-tuned model against the 112 gold
reviews never used for weight updates (validation + test). Training-slice reviews are excluded,
since errors there would understate real generalization failure.

**Overall error rate: 13.4%** (90 of 672 aspect-level judgments).

**Errors by aspect:** `food_taste` 25 · `service` 25 · `ambiance` 14 · `portion_size` 11 ·
`price` 10 · `authenticity` 5

`food_taste` and `service` are still the weakest — **the same two aspects that were weakest in
the raw weak labels, the baseline, the fine-tuned model, and now the gold-tuned model.** That
consistency across four independent stages indicates these aspects are intrinsically harder to
detect, not an artifact of any single training run.

**Two opposite failure modes:**

| Confusion | n | Interpretation |
|---|---|---|
| `not_mentioned` → `positive` | 32 | The **inherited over-triggering bias** from the weak labeler — projecting overall tone onto salient aspects. Reduced by gold fine-tuning, not eliminated. |
| `positive` → `not_mentioned` | 16 | The **opposite** error: missing brief single-clause mentions. Concentrated in `price`, where sentiment is often a short aside (*"great value for money"*) buried in a longer review. |

**Length has a concrete technical cause.** Misclassified reviews average **131 words versus 91**
for the eval set. The pipeline truncates at **256 tokens (~197 words)**, and **11% of the corpus
exceeds that.** Aspect mentions late in a long review may never reach the model at all — which
also explains the `price → not_mentioned` cluster, since price comments often come last. This is
a fixable configuration limit, not an inherent model weakness.

### The gold labels are not perfectly clean either

Reading the misclassifications surfaced likely annotation errors, not model errors:

- *"Just ok. Nothing special going on here except the service which is excellent! Food I'd
  mediocre at best"* → gold: `food_taste: not_mentioned`. The review explicitly rates the food.
- *"Hands down the best Indian food in STL… Each and every item on the buffet was outstanding"*
  → gold: `food_taste: neutral`. That is unambiguously positive.

Roughly 2 of 10 sampled errors look closer to labeling noise than model failure. **Reported
accuracy is therefore likely a slight underestimate.** Stated here rather than treating a
single-annotator gold set as infallible ground truth — and it argues for multi-annotator
labeling with measured inter-annotator agreement in any future iteration.

---

## 11. Deployment

### Architecture

```
Browser
  → reviews.dhakakacchi.com  (ACM cert, DNS via Hostinger)
  → API Gateway HTTP API
  → AWS Lambda (container image, 3008MB)
  → FastAPI + Mangum
      ├── GET  /          → one-page frontend
      ├── GET  /health    → health check
      └── POST /predict   → six-aspect sentiment + confidence
```

`deploy/` — FastAPI serves both the API and the single-page frontend, so there is no separate
frontend host to deploy, secure, or keep in sync.

### Honest uncertainty: the confidence threshold

Testing the deployed model on a Dhaka-Kacchi-style review — *"The biryani tastes good but the
price is very high. their service is not good"* — produced two wrong predictions:
`food_taste: not_mentioned` (should be positive) and `price: neutral` (should be negative).

**The confidence scores were the interesting part.** Both wrong predictions sat at **54–55%**,
barely above a four-class coin flip, while the correct ones were at 91% and 96%. The model was
signalling its own uncertainty; the UI was hiding it by rendering every prediction with equal
authority.

Fix: predictions below **0.65 confidence** are surfaced as `uncertain` rather than asserted.
`not_mentioned` is exempt, since a low-confidence "not mentioned" remains a reasonable default.
Threshold chosen from the error analysis, where misclassifications cluster in the 50–55% band.

Two contributing causes for the miss itself, both known: "biryani" is **underrepresented
vocabulary** (Bangladeshi cuisine had 1 business in the whole Yelp corpus and was filtered out),
and this is exactly the documented `positive → not_mentioned` failure mode, live.

### Container decisions

- **CPU-only PyTorch** (`torch==2.6.0+cpu`) — the CUDA build is several GB larger and useless for
  single-review CPU inference
- **Model weights, tokenizer, and config bundled into the image** — the app calls
  `from_pretrained` on local paths, so there are **no Hugging Face Hub requests at runtime**:
  faster cold starts, no external dependency, works under restricted egress
- **`AutoModel.from_config`, not `from_pretrained`** — the fine-tuned `state_dict` overwrites the
  base weights immediately, so downloading 250MB of pretrained weights to discard them is pure
  cold-start latency
- **Dependencies installed before app code** — `requirements.txt` is its own layer, so code
  changes rebuild in seconds instead of reinstalling PyTorch
- **Non-root user** and a container `HEALTHCHECK`
- **Pinned versions throughout** — reproducible builds

### Deployment challenges

**Platform mismatch.** The image built on Apple Silicon is ARM64; AWS Lambda runs x86_64.
Required `--platform linux/amd64` explicitly. Silent until the deploy fails.

**`torch==2.14.0+cpu` did not exist** for this platform on the PyTorch CPU index — pinned to
`2.6.0+cpu`, the newest available. Model weights load fine across these versions.

**numpy tried to build from source.** pip resolved numpy 2.4.6, which ships no prebuilt wheel for
this image and needs a C compiler the slim Lambda base image does not have. Pinned
`numpy==1.26.4`, which has a wheel. A dependency you never named directly can still break the
build.

**Buildx attestations broke Lambda.** `create-function` rejected the image:
*"manifest, config or layer media type … is not supported."* Modern Buildx attaches
provenance/SBOM metadata by default, producing a manifest list Lambda's validator refuses.
Fixed with `--provenance=false --sbom=false`.

**App Runner was blocked entirely.** The original plan was AWS App Runner — a persistent
container, no cold starts, the conventional choice for a steadily-used API. The AWS account
returned `SubscriptionRequiredException`: a new-account "free plan" restriction gating managed
compute services until identity/payment verification completes.

Pivoted to **Lambda**, already proven on the same account from a prior project. This required
adding **Mangum** (an ASGI-to-Lambda adapter) and a Lambda-specific Dockerfile — but **zero
changes to the FastAPI application code**, which is precisely the payoff for having written a
standard ASGI app rather than something framework-coupled.

For a low-traffic portfolio demo, Lambda's scale-to-zero pricing is arguably the *better* fit —
near-zero cost when idle, versus paying for an always-on container. The trade-off is cold starts.

**Lambda Function URLs returned 403 despite a correct policy.** `AuthType: NONE` was set and the
resource policy matched AWS's own documented example, verified in both the CLI and the console.
Still `AccessDeniedException`.

Debugging path: CloudTrail Event History showed nothing (it logs control-plane calls, not
data-plane HTTP traffic). A hypothesized per-function public-access-block setting turned out not
to exist in the current CLI or console. **The decisive test was invoking the function directly:**

```
aws lambda invoke → StatusCode: 200
```

The function, its IAM role, and the model all worked. The problem was **scoped to Function URLs
specifically.** Rather than continue guessing at an account-level restriction, the fix was to
route around it: **API Gateway HTTP API**, a different permission model — which worked
immediately, with no code changes, because API Gateway's event format is what Mangum already
expects natively.

**Cold starts exceeded the API Gateway timeout.** At 2048MB, every cold invocation hit exactly
`30000ms` and was killed: `INIT_REPORT … Status: timeout`. Loading PyTorch, transformers, and the
model genuinely took longer than that.

The trap: raising Lambda's own timeout does not help, because **API Gateway HTTP APIs impose a
separate, non-configurable ~29s integration timeout.** The cold start had to get *faster*, not be
given more time.

Because **Lambda scales CPU with memory**, raising memory to **3008MB** sped up the CPU-bound
import and load work enough to land under the ceiling. Peak memory used was only ~1.3GB — the
extra memory was bought for CPU, not for RAM. Warm requests then run in single-digit seconds.

### Custom domain

ACM certificate for `reviews.dhakakacchi.com` (DNS-validated via a CNAME at Hostinger), API
Gateway native custom domain with `EndpointType=REGIONAL`, API mapped to the `$default` stage,
and a final CNAME pointing `reviews` at the API Gateway target hostname. TLS 1.2 minimum.

### Cloud security practices

- **Two separate least-privilege IAM roles**, each scoped to one job: an ECR-pull role, and a
  Lambda execution role limited to CloudWatch Logs
- **API Gateway invoke permission scoped by `SourceArn`** to the specific API ID — not a wildcard
  principal
- No credentials in the image; `.env` gitignored, with a committed `.env.example` template
- Billing alerts enabled before deploying anything that bills by usage

---

## Engineering practices applied

**Data**
- Every dataset inspected before use; two rejected on inspection, and documented
- Per-business review caps to prevent single-entity domination
- Deterministic, committed preprocessing output
- Category ambiguity resolved by reading the data, not trusting tags

**Experimentation**
- **No train/test leakage** — the gold set is excluded from weak-label training entirely; when
  gold fine-tuning required using gold for training, it was re-split with a genuinely held-out
  slice
- **Seeded and reproducible** — `random_state=42` for sampling and splits; the error analysis
  reconstructs the exact fine-tuning split rather than re-shuffling
- **A real baseline before a neural model**, used as a decision gate rather than a formality
- **Macro F1 reported alongside accuracy**, because accuracy alone is misleading under this class
  imbalance — and reporting only accuracy would have hidden the project's central finding
- **A falsifiable hypothesis, then an experiment** to test it (label noise as the ceiling)
- Early stopping on validation, never on test
- Errors read individually, not just aggregated

**Labeling**
- Weak-label quality **measured**, not assumed
- Errors decomposed (detection vs. polarity) to localize failure rather than just score it
- Schema validation at write time, so bad labels never enter the dataset
- A purpose-built tool to make 1,914 human judgments actually finishable

**Production**
- Idempotent, resumable long-running jobs
- Graceful degradation: recursive batch splitting, adaptive rate-limit backoff, retries
- Rate limits respected rather than circumvented
- Pinned dependencies; minimal, layer-cached, non-root images
- Health checks at both container and application level
- **Uncertainty surfaced rather than hidden** from users
- Least-privilege IAM throughout

---

## Known limitations

1. **Cold starts.** First request after idle takes several seconds. **Provisioned Concurrency** is
   the standard fix, at a small ongoing cost.
2. **256-token truncation** drops content in 11% of reviews and demonstrably causes errors.
3. **Domain mismatch for the actual business.** Bangladeshi cuisine had 1 business in the entire
   Yelp corpus and was filtered out. The model is measurably less certain on biryani-specific
   vocabulary.
4. **Single-annotator gold set**, with identified annotation errors. No inter-annotator agreement
   measured.
5. **Weak labels from two models** — ~4% Groq `gpt-oss-120b`, ~96% local `gpt-oss:20b`.
6. **Persistent over-triggering.** `not_mentioned → positive` remains the dominant error,
   inherited from the weak labeler.
7. **`neutral` remains weak** across every aspect and every model — it is rare in the data and
   genuinely ambiguous for human annotators too.
8. **Convenience sample**, not uniform: 428 of 4,867 eligible businesses, English-only, US/Canada,
   2005–2020.

**Next steps, in rough priority order:** multi-annotator gold labeling with measured agreement ·
longer context or a long-context model · fine-tuning on real Dhaka Kacchi reviews once volume
allows · German-language support for the Berlin market (GERestaurant is the obvious starting
point) · Provisioned Concurrency · monitoring for prediction drift.

---

## Reproducing this project

```bash
git clone https://github.com/shadmanArko/dhaka-kacchi-review-intelligence
cd dhaka-kacchi-review-intelligence
uv sync
cp .env.example .env   # fill in GROQ_API_KEY if using src/label_reviews_groq.py
```

All scripts below use paths relative to `src/`, so `cd src` first (or prefix each
command with `uv run --directory src`).

**Data** — download the [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/)
into `data/raw/`, then, from `src/`:

```bash
cd src
uv run python filter_reviews.py ../data/raw/yelp_academic_dataset_review.json
# → writes data/interim/filtered_reviews.json (preprocessing into
#   data/processed/data_processed.{csv,json} is done separately; see that
#   folder's committed output if you want to skip straight to labeling/training)
```

**Weak labeling** — requires [Ollama](https://ollama.com) and `ollama pull gpt-oss:20b`
(~2.8 days unattended, fully resumable):

```bash
uv run python label_reviews_ollama.py
```

**Gold labeling** — opens at `http://localhost:5050`:

```bash
uv run python labeling_app.py
```

**Train and evaluate:**

```bash
uv run python baseline_model.py       # TF-IDF + LogReg
uv run python finetune_multihead.py   # ~34 min on an M3 Pro
uv run python gold_finetune.py        # few minutes
uv run python error_analysis.py
```

**Serve locally:**

```bash
cd ../deploy
mkdir -p model_weights
cp ../model_multihead_goldtuned/* model_weights/
docker build -f Dockerfile.lambda -t review-intelligence .
docker run -p 8080:8080 review-intelligence   # → http://localhost:8080
```

---

## Repository layout

```
├── src/
│   ├── filter_reviews.py         # streaming review extraction from the raw Yelp dump
│   ├── label_reviews_groq.py     # weak labeling via Groq (superseded)
│   ├── label_reviews_ollama.py   # weak labeling via local Ollama
│   ├── labeling_app.py           # local Flask labeling UI for the gold set
│   ├── baseline_model.py         # TF-IDF + Logistic Regression
│   ├── finetune_multihead.py     # multi-head DistilBERT
│   ├── gold_finetune.py          # gold-label fine-tuning experiment
│   └── error_analysis.py         # error extraction and breakdown
├── deploy/
│   ├── app/                      # FastAPI backend
│   ├── static/index.html         # one-page frontend
│   ├── model_weights/            # gitignored; populated at build time (see above)
│   ├── Dockerfile.lambda         # AWS Lambda build
│   ├── .dockerignore
│   └── requirements.txt
├── data/
│   ├── raw/                      # gitignored; place the downloaded Yelp dump here
│   ├── interim/                  # gitignored; regenerable filter output
│   └── processed/                # committed cleaned dataset + weak/gold labels
├── model_multihead/               # gitignored; output of finetune_multihead.py
├── model_multihead_goldtuned/     # gitignored; output of gold_finetune.py — the
│                                  # production model, mirrored on Hugging Face Hub
├── results/                      # metrics, charts, error dumps
└── notebooks/                    # exploratory analysis
```

**Model:** [`shadmanArko/dhaka-kacchi-review-intelligence`](https://huggingface.co/shadmanArko/dhaka-kacchi-review-intelligence)
on Hugging Face Hub — the gold-fine-tuned production model (see [section 9](#9-breaking-the-label-noise-ceiling)).
