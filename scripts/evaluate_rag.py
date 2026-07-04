"""Small, local-only retrieval quality check for the SkinSense knowledge base."""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Evaluation must use an already-cached embedding model and never reach the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from rag.retrieve import IndexNotReadyError, RetrievalError, get_collection, retrieve_rules


EVALUATION_CASES: List[Dict[str, object]] = [
    {
        "name": "Oily, acne-prone skin",
        "query": "Skin type: oily. Key concerns: acne and clogged-feeling skin.",
        "expected": ("skin_type:oily", "concern:acne"),
    },
    {
        "name": "Dry, sensitive skin",
        "query": "Skin type: dry. Key concerns: dryness and sensitivity. Keep it gentle.",
        "expected": ("skin_type:dry", "concern:dryness", "skin_type:sensitive"),
    },
    {
        "name": "Pigmentation",
        "query": "Skin type: combination. Key concerns: pigmentation and uneven-looking tone.",
        "expected": ("concern:pigmentation",),
    },
    {
        "name": "Redness",
        "query": "Skin type: sensitive. Key concerns: visible redness and discomfort.",
        "expected": ("concern:redness", "skin_type:sensitive"),
    },
    {
        "name": "Sunscreen",
        "query": "I need a comfortable morning sunscreen step and sun protection.",
        "expected": ("concern:sun_protection", "routine:am"),
    },
    {
        "name": "Beginner routine",
        "query": (
            "What step order should a beginner use for a gentle morning and evening "
            "routine with cleanser, serum, moisturizer, and sunscreen?"
        ),
        "expected": ("concern:any", "routine:am_step_order", "routine:pm_step_order"),
    },
]


def evaluate(k: int) -> int:
    """Print retrieval evidence and return the number of cases needing review."""
    collection = get_collection()
    review_count = 0

    for case in EVALUATION_CASES:
        results = retrieve_rules(collection, str(case["query"]), k=k)
        expected = tuple(case["expected"])
        retrieved_tags = " ".join(
            str(result.get("metadata", {}).get("tags", "")) for result in results
        )
        matched = [tag for tag in expected if tag in retrieved_tags]
        passed = bool(matched)
        review_count += int(not passed)

        print(f"\n[{ 'PASS' if passed else 'REVIEW' }] {case['name']}")
        print(f"Query: {case['query']}")
        print(f"Expected tag match (any): {', '.join(expected)}")
        print(f"Matched: {', '.join(matched) if matched else 'none'}")
        print("Retrieved:")
        for result in results:
            tags = result.get("metadata", {}).get("tags", "")
            print(
                f"  {result['id']}  distance={result['distance']:.4f}  tags={tags}"
            )

    print(
        f"\nSummary: {len(EVALUATION_CASES) - review_count}/"
        f"{len(EVALUATION_CASES)} cases matched an expected tag."
    )
    return review_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5, help="Retrieved cards per query.")
    args = parser.parse_args()
    try:
        review_count = evaluate(args.k)
    except (IndexNotReadyError, RetrievalError, ValueError) as exc:
        raise SystemExit(f"Evaluation could not run: {exc}") from exc
    raise SystemExit(1 if review_count else 0)


if __name__ == "__main__":
    main()
