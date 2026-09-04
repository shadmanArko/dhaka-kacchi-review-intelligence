"""
Weak-label restaurant reviews for aspect-based sentiment using Groq.

Features:
- Reads GROQ_API_KEY from .env
- Explicitly overrides an existing environment variable
- Resumable: skips reviews already present in labels_weak.jsonl
- Processes reviews in batches
- Validates every model response before writing
- Handles rate limits with exponential backoff
- Prints useful rate-limit information
- Distinguishes temporary rate limits from quota exhaustion
- Never prints the full API key

Setup:

1. Install dependencies:

    pip install groq python-dotenv

2. Create a .env file in the same folder:

    GROQ_API_KEY=gsk_your_key_here

3. Put data_processed.json in the same folder.

4. Run:

    python label_reviews_groq.py
"""

import json
import os
import re
import sys
import time

from dotenv import load_dotenv
from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "openai/gpt-oss-120b"

# Number of reviews sent in one API request.
BATCH_SIZE = 5

# Seconds between successful API calls.
SLEEP_BETWEEN_CALLS = 3

# Maximum retries for temporary rate limits.
MAX_RETRIES = 6

# Maximum output tokens.
MAX_OUTPUT_TOKENS = 1200

INPUT_FILE = "../data/processed/data_processed.json"
OUTPUT_FILE = "../data/processed/labels_weak.jsonl"


ASPECTS = [
    "food_taste",
    "service",
    "price",
    "portion_size",
    "authenticity",
    "ambiance",
]

SENTIMENTS = [
    "positive",
    "negative",
    "neutral",
    "not_mentioned",
]


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a precise data-labeling assistant for restaurant reviews.

For each review, determine the sentiment expressed toward each aspect.

Aspects:

- food_taste:
  quality or taste of the food itself

- service:
  staff friendliness, speed, attentiveness, or helpfulness

- price:
  value for money, whether the food/service felt expensive or cheap

- portion_size:
  whether the portion was generous, sufficient, or small

- authenticity:
  whether the food felt authentic, traditional, genuine, or inauthentic

- ambiance:
  atmosphere, decor, noise, cleanliness, comfort, or overall environment

For every aspect, output exactly ONE of:

"positive"
"negative"
"neutral"
"not_mentioned"

Rules:

1. Use "not_mentioned" when the review does not discuss the aspect.

2. Use "neutral" only when the aspect is actually mentioned but the
   sentiment is genuinely neutral or mixed.

3. Do not infer an aspect that is not discussed.

4. Do not invent information.

5. Return exactly one object for every review.

6. Keep the same review_id provided in the input.

Respond ONLY with valid JSON.

Required structure:

{
  "labels": [
    {
      "review_id": "<id>",
      "food_taste": "positive",
      "service": "not_mentioned",
      "price": "neutral",
      "portion_size": "positive",
      "authenticity": "not_mentioned",
      "ambiance": "positive"
    }
  ]
}
"""


# ============================================================
# ENVIRONMENT / API KEY
# ============================================================

def load_api_key():
    """
    Load GROQ_API_KEY from .env.

    override=True is intentional:
    if the shell already has an old GROQ_API_KEY,
    the .env value will replace it.
    """

    load_dotenv(override=True)

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        print(
            "\nERROR: GROQ_API_KEY was not found.\n"
            "Create a .env file containing:\n\n"
            "GROQ_API_KEY=gsk_your_key_here\n"
        )
        sys.exit(1)

    return api_key


def print_key_info(api_key):
    """
    Print safe information about the loaded API key.
    Never print the complete key.
    """

    if len(api_key) >= 12:
        prefix = api_key[:8]
        suffix = api_key[-4:]

        print(f"API key loaded successfully.")
        print(f"Key fingerprint: {prefix}...{suffix}")
    else:
        print("API key loaded, but key appears unusually short.")


# ============================================================
# FILE HANDLING
# ============================================================

def load_reviews():
    """Load input reviews from JSON."""

    if not os.path.exists(INPUT_FILE):
        print(f"\nERROR: Input file not found: {INPUT_FILE}")
        sys.exit(1)

    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            reviews = json.load(f)
    except json.JSONDecodeError as e:
        print(f"\nERROR: Could not parse {INPUT_FILE}")
        print(e)
        sys.exit(1)

    if not isinstance(reviews, list):
        print("\nERROR: data_processed.json must contain a JSON list.")
        sys.exit(1)

    return reviews


def load_already_labeled():
    """
    Read existing JSONL output and return review IDs that
    have already been successfully labeled.
    """

    done = set()

    if not os.path.exists(OUTPUT_FILE):
        return done

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)

                review_id = obj.get("review_id")

                if review_id is not None:
                    done.add(str(review_id))

            except json.JSONDecodeError:
                print(
                    f"Warning: ignoring invalid JSON on "
                    f"line {line_number} of {OUTPUT_FILE}"
                )

    return done


# ============================================================
# PROMPT BUILDING
# ============================================================

def build_user_prompt(batch):
    """
    Build the user prompt for a batch.

    Reviews are truncated to 1200 characters to control
    token usage.
    """

    lines = []

    for review in batch:

        review_id = str(review.get("review_id", ""))

        text = str(review.get("text", ""))

        # Prevent enormous reviews from consuming the quota.
        text = text[:1200]

        lines.append(
            f"review_id: {review_id}\n"
            f"text: {json.dumps(text, ensure_ascii=False)}"
        )

    return (
        "Label the following restaurant reviews.\n\n"
        + "\n\n".join(lines)
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_label(obj):
    """Validate one labeled review."""

    if not isinstance(obj, dict):
        return False

    if "review_id" not in obj:
        return False

    for aspect in ASPECTS:

        if obj.get(aspect) not in SENTIMENTS:
            return False

    return True


def validate_batch(labels, batch):
    """
    Make sure the model returned exactly one valid label
    for every review in the batch.

    We require exact ID matching.
    """

    if not isinstance(labels, list):
        return False, "labels is not a list"

    if len(labels) != len(batch):
        return (
            False,
            f"expected {len(batch)} labels but received {len(labels)}",
        )

    expected_ids = [
        str(review["review_id"])
        for review in batch
    ]

    returned_ids = [
        str(obj.get("review_id"))
        for obj in labels
        if isinstance(obj, dict)
    ]

    if expected_ids != returned_ids:
        return (
            False,
            f"review IDs/order mismatch. "
            f"Expected {expected_ids}, got {returned_ids}",
        )

    for obj in labels:

        if not validate_label(obj):
            return (
                False,
                f"invalid label object: {obj}",
            )

    return True, ""


# ============================================================
# RATE LIMIT INFORMATION
# ============================================================

def print_rate_limit_headers(response):
    """
    Print Groq rate-limit information when available.

    This helps distinguish:
    - requests remaining
    - tokens remaining
    - reset time
    """

    try:

        headers = response.headers

    except Exception:
        return

    interesting = [
        "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-reset-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-tokens",
    ]

    found = False

    print("\nRate-limit information:")

    for key in interesting:

        value = headers.get(key)

        if value is not None:

            print(f"  {key}: {value}")

            found = True

    if not found:
        print("  Rate-limit headers not available.")


# ============================================================
# ERROR ANALYSIS
# ============================================================

def extract_wait_time(message):
    """
    Try to extract Groq's suggested retry time.

    Handles examples such as:

        try again in 6m46s
        try again in 20s
        try again in 1m
    """

    patterns = [
        r"try again in (?:(\d+)m)?\s*(\d+(?:\.\d+)?)s",
        r"retry in (?:(\d+)m)?\s*(\d+(?:\.\d+)?)s",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            message,
            flags=re.IGNORECASE,
        )

        if match:

            minutes = int(match.group(1) or 0)

            seconds = float(match.group(2))

            return minutes * 60 + seconds

    return None


def classify_error(error):
    """
    Classify the error without assuming that every 429
    means a daily quota.
    """

    message = str(error)

    lower = message.lower()

    is_429 = (
        "429" in lower
        or "rate_limit" in lower
        or "rate limit" in lower
    )

    is_daily = (
        "tpd" in lower
        or "rpd" in lower
        or "tokens per day" in lower
        or "requests per day" in lower
        or "daily" in lower
    )

    wait_time = extract_wait_time(message)

    return {
        "message": message,
        "is_429": is_429,
        "is_daily": is_daily,
        "wait_time": wait_time,
    }


# ============================================================
# API CALL
# ============================================================

def label_batch(client, batch, batch_number, total_batches):

    user_prompt = build_user_prompt(batch)

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            print(
                f"\nBatch {batch_number}/{total_batches} "
                f"| attempt {attempt}/{MAX_RETRIES}"
            )

            response = client.chat.completions.create(

                model=MODEL,

                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],

                temperature=0,

                max_tokens=MAX_OUTPUT_TOKENS,

                response_format={
                    "type": "json_object"
                },

                reasoning_effort="low",
            )

            # Show rate-limit information if available.
            print_rate_limit_headers(response)

            raw = response.choices[0].message.content.strip()

            # Remove accidental markdown fences.
            if raw.startswith("```"):
                raw = re.sub(
                    r"^```(?:json)?\s*",
                    "",
                    raw,
                    flags=re.IGNORECASE,
                )

                raw = re.sub(
                    r"\s*```$",
                    "",
                    raw,
                )

            parsed = json.loads(raw)

            if not isinstance(parsed, dict):
                raise ValueError(
                    "Model response is not a JSON object."
                )

            labels = parsed.get("labels")

            valid, reason = validate_batch(
                labels,
                batch,
            )

            if not valid:

                print(
                    "\nMODEL OUTPUT VALIDATION FAILED:"
                )

                print(reason)

                print(
                    "\nRaw model output:"
                )

                print(raw[:2000])

                return None

            print(
                f"Batch {batch_number}/{total_batches}: "
                f"SUCCESS — {len(labels)} reviews labeled."
            )

            return labels

        except json.JSONDecodeError as e:

            print(
                "\nERROR: Model returned invalid JSON."
            )

            print(str(e))

            if "raw" in locals():
                print("\nRaw output:")
                print(raw[:2000])

            return None

        except Exception as e:

            info = classify_error(e)

            print(
                f"\nGROQ ERROR:\n{info['message']}"
            )

            # ------------------------------------------------
            # Temporary rate limit
            # ------------------------------------------------

            if info["is_429"]:

                if info["wait_time"] is not None:

                    wait_time = (
                        info["wait_time"] + 5
                    )

                else:

                    # Exponential backoff.
                    wait_time = min(
                        60,
                        5 * (2 ** (attempt - 1)),
                    )

                print(
                    f"\nRate limit encountered."
                )

                if info["is_daily"]:

                    print(
                        "The error message appears to "
                        "mention a daily/token quota."
                    )

                else:

                    print(
                        "This does NOT necessarily mean "
                        "the daily quota is exhausted."
                    )

                if attempt < MAX_RETRIES:

                    print(
                        f"Waiting {wait_time:.0f} seconds "
                        f"before retry..."
                    )

                    time.sleep(wait_time)

                    continue

                print(
                    "\nMaximum retries reached."
                )

                return None

            # ------------------------------------------------
            # Unknown error
            # ------------------------------------------------

            print(
                "\nNon-rate-limit error. "
                "This batch will not be written."
            )

            return None

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("GROQ WEAK-LABELING SCRIPT")
    print("=" * 60)

    # --------------------------------------------------------
    # API key
    # --------------------------------------------------------

    api_key = load_api_key()

    print_key_info(api_key)

    # --------------------------------------------------------
    # Create Groq client
    # --------------------------------------------------------

    client = Groq(
        api_key=api_key
    )

    # --------------------------------------------------------
    # Load reviews
    # --------------------------------------------------------

    reviews = load_reviews()

    print(
        f"\nTotal reviews in input: {len(reviews)}"
    )

    # --------------------------------------------------------
    # Find already processed reviews
    # --------------------------------------------------------

    already_done = load_already_labeled()

    remaining = [
        review
        for review in reviews
        if str(review.get("review_id"))
        not in already_done
    ]

    print(
        f"Already labeled: {len(already_done)}"
    )

    print(
        f"Remaining: {len(remaining)}"
    )

    # --------------------------------------------------------
    # Nothing to do
    # --------------------------------------------------------

    if not remaining:

        print(
            "\nEverything has already been labeled."
        )

        return

    # --------------------------------------------------------
    # Create batches
    # --------------------------------------------------------

    batches = [
        remaining[i:i + BATCH_SIZE]
        for i in range(
            0,
            len(remaining),
            BATCH_SIZE,
        )
    ]

    print(
        f"Batches to process: {len(batches)}"
    )

    # --------------------------------------------------------
    # Open output file
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "a",
        encoding="utf-8",
    ) as out_f:

        for batch_number, batch in enumerate(
            batches,
            start=1,
        ):

            labels = label_batch(
                client=client,
                batch=batch,
                batch_number=batch_number,
                total_batches=len(batches),
            )

            # ------------------------------------------------
            # Failed batch
            # ------------------------------------------------

            if labels is None:

                print(
                    f"\nBatch {batch_number} was NOT written."
                )

                print(
                    "Stopping so you can inspect the "
                    "error and safely resume later."
                )

                break

            # ------------------------------------------------
            # Write successful labels
            # ------------------------------------------------

            for label in labels:

                out_f.write(
                    json.dumps(
                        label,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            out_f.flush()

            print(
                f"Saved {len(labels)} labels "
                f"to {OUTPUT_FILE}"
            )

            # ------------------------------------------------
            # Pause between requests
            # ------------------------------------------------

            if batch_number < len(batches):

                time.sleep(
                    SLEEP_BETWEEN_CALLS
                )

    print("\n" + "=" * 60)
    print("RUN FINISHED")
    print("=" * 60)

    print(
        "Run the script again later to resume. "
        "Already-labeled reviews will be skipped."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()