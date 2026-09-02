"""
Weak-labels reviews for aspect-based sentiment using a LOCAL model via
Ollama (gpt-oss:20b) - no API key, no rate limits, no cost. Runs entirely
on your own machine.

Setup:
    1. Install Ollama: https://ollama.com/download (Mac app or `brew install ollama`)
    2. Pull the model:  ollama pull gpt-oss:20b
    3. Make sure Ollama is running (it starts automatically after install,
       or run `ollama serve` manually)
    4. pip install openai   (we use the OpenAI-compatible client Ollama exposes)
    5. python label_reviews_ollama.py

Resumable: re-running skips reviews already labeled in the output file.

Input:  data_processed.json  (from the preprocessing step)
Output: labels_weak.jsonl    (one JSON object per review, appended as we go)
"""
import json
import os
import time
from openai import OpenAI

MODEL = "gpt-oss:20b"
BATCH_SIZE = 6            # smaller than the Groq version - local reasoning + generation is
                          # slower per-token, so smaller batches finish more reliably
INPUT_FILE = "data_processed.json"
OUTPUT_FILE = "labels_weak.jsonl"
MAX_RETRIES_PER_BATCH = 3
MAX_TOKENS = 6000          # generous headroom - reasoning models can burn a lot of tokens
                           # "thinking" before the actual answer; no cost concern running locally

ASPECTS = ["food_taste", "service", "price", "portion_size", "authenticity", "ambiance"]
SENTIMENTS = ["positive", "negative", "neutral", "not_mentioned"]

SYSTEM_PROMPT = f"""You are a precise data-labeling assistant for restaurant reviews.

For each review, determine the sentiment expressed toward each of these aspects:
- food_taste: quality/taste of the food itself
- service: staff friendliness, speed, attentiveness
- price: value for money, whether it felt expensive/cheap
- portion_size: whether portions were generous or small
- authenticity: whether the food felt authentic/traditional vs. inauthentic
- ambiance: atmosphere, decor, noise, cleanliness of the space

For each aspect, output exactly one of: "positive", "negative", "neutral", "not_mentioned".
Use "not_mentioned" if the review does not discuss that aspect at all.
Use "neutral" only if the aspect is mentioned but sentiment is genuinely mixed/neutral.

Respond with ONLY a JSON object, no other text, no markdown fences, shaped exactly like:
{{"labels": [{{"review_id": "<id>", "food_taste": "...", "service": "...", "price": "...", "portion_size": "...", "authenticity": "...", "ambiance": "..."}}, ...]}}
One entry per review, in the same order given.
"""

def build_user_prompt(batch):
    lines = []
    for r in batch:
        text = r["text"][:1200]
        lines.append(f'review_id: {r["review_id"]}\ntext: "{text}"')
    return "Label these reviews:\n\n" + "\n\n".join(lines)


def load_already_labeled():
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


def validate_label(obj):
    if "review_id" not in obj:
        return False
    for aspect in ASPECTS:
        if obj.get(aspect) not in SENTIMENTS:
            return False
    return True


def call_model(client, batch):
    """Sends one batch to the model, returns (valid_labels, success_bool)."""
    user_prompt = build_user_prompt(batch)
    raw = None
    finish_reason = None
    for attempt in range(MAX_RETRIES_PER_BATCH):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},
                extra_body={"options": {"num_ctx": 12000}},
            )
            raw = resp.choices[0].message.content.strip()
            finish_reason = resp.choices[0].finish_reason
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(raw)
            labels = parsed["labels"] if isinstance(parsed, dict) else parsed
            valid = [obj for obj in labels if validate_label(obj)]
            return valid, True

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"    parse failed ({e}), finish_reason={finish_reason}, "
                  f"attempt {attempt+1}/{MAX_RETRIES_PER_BATCH}")

        except Exception as e:
            msg = str(e)
            if "connection" in msg.lower() or "refused" in msg.lower():
                raise ConnectionError("Can't connect to Ollama. Is it running? Try: ollama serve")
            print(f"    error - {msg}, attempt {attempt+1}/{MAX_RETRIES_PER_BATCH}")
            time.sleep(3)

    return [], False


def label_batch_with_splitting(client, batch, depth=0):
    """Tries a batch; if it keeps failing (usually truncation), splits it in
    half and retries each half - down to single reviews if needed - instead
    of losing the whole batch."""
    valid, success = call_model(client, batch)
    if success:
        return valid

    if len(batch) == 1:
        print(f"    giving up on review {batch[0]['review_id']} - unfixable at single-review size")
        return []

    print(f"    batch of {len(batch)} kept failing - splitting in half and retrying")
    mid = len(batch) // 2
    left = label_batch_with_splitting(client, batch[:mid], depth + 1)
    right = label_batch_with_splitting(client, batch[mid:], depth + 1)
    return left + right


def main():
    # Ollama exposes an OpenAI-compatible endpoint locally - no real API key needed
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    with open(INPUT_FILE, encoding="utf-8") as f:
        reviews = json.load(f)

    already_done = load_already_labeled()
    remaining = [r for r in reviews if r["review_id"] not in already_done]
    print(f"Total reviews: {len(reviews)} | already labeled: {len(already_done)} | remaining: {len(remaining)}")

    if not remaining:
        print("Nothing left to label.")
        return

    out_f = open(OUTPUT_FILE, "a", encoding="utf-8")
    batches = [remaining[i:i + BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]
    start_time = time.time()

    for i, batch in enumerate(batches):
        try:
            valid_labels = label_batch_with_splitting(client, batch)
        except ConnectionError as e:
            print(f"ERROR: {e}")
            out_f.close()
            return

        for obj in valid_labels:
            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        out_f.flush()

        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed * 60  # batches per minute
        print(f"Batch {i+1}/{len(batches)}: labeled {len(valid_labels)}/{len(batch)} reviews "
              f"({rate:.1f} batches/min)")

    out_f.close()
    total_min = (time.time() - start_time) / 60
    print(f"Done. Processed {len(batches)} batches in {total_min:.1f} minutes.")


if __name__ == "__main__":
    main()