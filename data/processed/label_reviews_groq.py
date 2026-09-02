"""
Weak-labels reviews for aspect-based sentiment using Groq's free API
(GPT-OSS-120B). Processes reviews in batches, with adaptive backoff on
rate limits and a clean stop/resume on daily caps.

Resumable: re-running skips reviews already labeled in the output file,
so if you hit the daily cap partway through, just run it again tomorrow.

Setup:
    1. Get a free API key at https://console.groq.com
    2. pip install groq python-dotenv
    3. Create a .env file in this directory with: GROQ_API_KEY=your-key-here
    4. python label_reviews_groq.py

Input:  data_processed.json  (from the preprocessing step)
Output: labels_weak.jsonl    (one JSON object per review, appended as we go)
"""
import json
import os
import re
import time
import sys
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # reads .env in the current directory (or nearest parent) into os.environ

MODEL = "openai/gpt-oss-120b"
BATCH_SIZE = 10          # reviews per API call - kept modest since free-tier TPM/TPD for
                         # this model varies by source; adjust down if you still hit 429s often
SLEEP_BETWEEN_CALLS = 5   # base seconds between calls - the adaptive backoff below handles the rest
MAX_RETRIES_PER_BATCH = 5
MAX_DAILY_CAP_RETRIES = 30  # daily/rolling token cap can require several short waits in a row
INPUT_FILE = "data_processed.json"
OUTPUT_FILE = "labels_weak.jsonl"

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
        # truncate very long reviews to keep token usage predictable
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


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not found. Set it in a .env file (GROQ_API_KEY=...) "
              "in this directory, or export it as an environment variable.")
        sys.exit(1)

    client = Groq(api_key=api_key)

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

    for i, batch in enumerate(batches):
        user_prompt = build_user_prompt(batch)
        attempt = 0
        daily_cap_attempt = 0
        raw = None
        while True:
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                    max_tokens=1200,
                    response_format={"type": "json_object"},
                    reasoning_effort="low",
                )
                raw = resp.choices[0].message.content.strip()
                raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                parsed = json.loads(raw)
                labels = parsed["labels"] if isinstance(parsed, dict) else parsed

                valid_count = 0
                for obj in labels:
                    if validate_label(obj):
                        out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        valid_count += 1
                out_f.flush()
                print(f"Batch {i+1}/{len(batches)}: labeled {valid_count}/{len(batch)} reviews")
                break  # success, move to next batch

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"Batch {i+1}/{len(batches)}: FAILED to parse model output ({e}), skipping batch")
                print(f"  Raw output was: {(raw or '')[:300]!r}")
                break

            except Exception as e:
                msg = str(e)
                is_rate_limit = "429" in msg or "rate_limit" in msg.lower()
                is_daily_cap = "rpd" in msg.lower() or "tpd" in msg.lower() or "daily" in msg.lower()

                if is_daily_cap:
                    # Groq tells us exactly how long to wait (e.g. "try again in 6m46s") -
                    # this is a rolling window, not a fixed midnight reset, so waiting it
                    # out is usually much faster than "come back tomorrow".
                    wait_match = re.search(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s", msg)
                    if wait_match and daily_cap_attempt < MAX_DAILY_CAP_RETRIES:
                        minutes = int(wait_match.group(1) or 0)
                        seconds = float(wait_match.group(2))
                        wait_time = minutes * 60 + seconds + 10  # +10s safety buffer
                        daily_cap_attempt += 1
                        print(f"Batch {i+1}/{len(batches)}: daily token cap hit. "
                              f"Waiting {wait_time:.0f}s as instructed, then retrying "
                              f"(attempt {daily_cap_attempt}/{MAX_DAILY_CAP_RETRIES})...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Batch {i+1}/{len(batches)}: hit daily cap and couldn't parse "
                              f"a wait time, or retries exhausted ({msg}).")
                        print("Stopping - re-run this script later to resume.")
                        out_f.close()
                        return

                if is_rate_limit:
                    attempt += 1
                    if attempt > MAX_RETRIES_PER_BATCH:
                        print(f"Batch {i+1}/{len(batches)}: rate limit retries exhausted, skipping batch")
                        break
                    backoff = min(60, 5 * (2 ** attempt))  # exponential backoff, capped at 60s
                    print(f"Batch {i+1}/{len(batches)}: rate limited, backing off {backoff}s "
                          f"(retry {attempt}/{MAX_RETRIES_PER_BATCH})")
                    time.sleep(backoff)
                    continue

                # Unknown error - log and skip this batch rather than loop forever
                print(f"Batch {i+1}/{len(batches)}: error - {msg}")
                break

        time.sleep(SLEEP_BETWEEN_CALLS)

    out_f.close()
    print("Done for this run. Re-run anytime to continue where it left off.")


if __name__ == "__main__":
    main()