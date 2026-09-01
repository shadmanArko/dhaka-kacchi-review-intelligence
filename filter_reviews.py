"""
Filters yelp_academic_dataset_review.json down to reviews for a
pre-selected set of businesses (see matched_businesses.json).

Usage:
    python filter_reviews.py /path/to/yelp_academic_dataset_review.json

Produces: filtered_reviews.json  (JSON array, ready to upload)
"""
import json
import sys
from collections import defaultdict

MAX_PER_BUSINESS = 150
MAX_TOTAL = 20000

def main(review_path):
    with open("matched_businesses.json") as f:
        matched = json.load(f)

    # Prioritize open businesses first
    matched.sort(key=lambda b: (b["is_open"] == 0))
    biz_lookup = {b["business_id"]: b for b in matched}
    target_ids = set(biz_lookup.keys())

    per_business_count = defaultdict(int)
    kept = []

    with open(review_path, encoding="utf-8") as f:
        for line in f:
            if len(kept) >= MAX_TOTAL:
                break
            review = json.loads(line)
            bid = review.get("business_id")
            if bid not in target_ids:
                continue
            if per_business_count[bid] >= MAX_PER_BUSINESS:
                continue
            biz = biz_lookup[bid]
            kept.append({
                "review_id": review["review_id"],
                "business_id": bid,
                "business_name": biz["name"],
                "city": biz["city"],
                "state": biz["state"],
                "categories": biz["categories"],
                "stars": review["stars"],
                "date": review["date"],
                "text": review["text"],
            })
            per_business_count[bid] += 1

    with open("filtered_reviews.json", "w", encoding="utf-8") as out:
        json.dump(kept, out, ensure_ascii=False)

    print(f"Kept {len(kept)} reviews across {len(per_business_count)} businesses")
    print("Saved to filtered_reviews.json")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python filter_reviews.py /path/to/yelp_academic_dataset_review.json")
        sys.exit(1)
    main(sys.argv[1])