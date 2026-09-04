## Error Analysis (Phase 8)

The gold fine-tuned model (Phase 7.5) was evaluated on the 112 gold-labeled reviews it never
used for weight updates (val + test splits), and every misclassification was extracted and
read individually to look for patterns beyond the aggregate metrics.

**Overall error rate: 13.4%** (90 of 672 aspect-level judgments), consistent with the 87.2%
accuracy reported in Phase 7.5.

### Persistent weak spots

`food_taste` and `service` remain the two hardest aspects (25 errors each) — the same two
that were weakest as far back as the Phase 5 weak-label agreement analysis. This consistency
across weak labels, baseline, fine-tuned, and gold-tuned models indicates these aspects are
intrinsically harder to detect, not an artifact of any single training stage.

### Two distinct, opposite failure modes

| Confusion | Count | Likely cause |
|---|---|---|
| `not_mentioned → positive` | 32 | Over-eager positivity projection — the model assumes the most salient topics (food, service) are positive when the review is upbeat overall, even if that topic isn't explicitly discussed. Same bias first documented in the weak labels (Phase 5), reduced but not eliminated by gold fine-tuning. |
| `positive → not_mentioned` | 16 | Missed brief, single-clause mentions — concentrated in `price`, where sentiment is often a short aside (e.g. *"great value for money"*) buried inside a longer review about food and atmosphere. |

### Longer reviews are measurably harder

Misclassified reviews average **131 words**, versus 91 words for the eval set overall. This
has a concrete technical explanation: the training pipeline truncates review text at 256
tokens (~197 words), and **11% of the full dataset exceeds that limit**. Aspect mentions that
occur late in a long review — such as a price comment near the end — may simply never reach
the model. This is a known, fixable limitation (e.g. increasing `MAX_LENGTH` or switching to
a model with longer context) rather than an inherent model weakness.

### Gold labels are not perfectly clean

Manual reading of misclassified examples surfaced at least two likely annotation
inconsistencies rather than genuine model errors — for example, a review stating *"Food I'd
mediocre at best"* was gold-labeled `food_taste: not_mentioned`. With a hand-labeled set of
this size, some fraction of reported "errors" reflects labeling noise rather than true model
failure, meaning reported accuracy is likely a slight underestimate of real model quality.
This is stated plainly here rather than treating the gold set as a perfect ground truth.

Full per-error data (all 90 mismatches with review text, true/predicted labels, cuisine, and
star rating): [`results/error_analysis.json`](results/error_analysis.json). Analysis code:
[`src/error_analysis.py`](src/error_analysis.py).
