"""
Generate a synthetic skincare rules knowledge base.

Each rule card has the shape:
{"id": "R001", "tags": "skin_type:oily concern:acne routine:am", "text": "..."}
"""

import json
import os
from typing import List, Dict


SKIN_TYPES = ["oily", "dry", "combination", "sensitive", "normal"]
CONCERNS = [
    "acne",
    "pigmentation",
    "dullness",
    "dryness",
    "redness",
    "texture",
    "fine_lines",
    "sun_protection",
]


def _base_rule_text(skin_type: str, concern: str, routine: str) -> str:
    """Create a short, cosmetic-only guideline text."""
    routine_label = "morning" if routine == "am" else "evening"
    parts = [
        f"For {skin_type} skin with {concern} concerns, build a gentle {routine_label} routine.",
        "Use mild, cosmetic skincare products and stop using anything that feels irritating.",
    ]

    if concern == "acne":
        parts.append(
            "Focus on lightweight, non-comedogenic textures and avoid heavy occlusive layers."
        )
    elif concern == "pigmentation":
        parts.append(
            "Look for brightening cosmetic ingredients and be consistent rather than using strong products."
        )
    elif concern == "dullness":
        parts.append(
            "Add gentle hydrating layers and avoid over-scrubbing the skin surface."
        )
    elif concern == "dryness":
        parts.append(
            "Use soft, cushioning textures and seal in hydration with a comfortable moisturizer."
        )
    elif concern == "redness":
        parts.append(
            "Choose simple, fragrance‑free formulas and keep water temperature lukewarm."
        )
    elif concern == "texture":
        parts.append(
            "Keep the barrier comfortable first; use any texture‑smoothing cosmetics sparingly."
        )
    elif concern == "fine_lines":
        parts.append(
            "Layer hydrating serums and moisturizers to reduce the look of fine lines caused by dryness."
        )
    elif concern == "sun_protection":
        parts.append(
            "Use a comfortable, broad cosmetic sunscreen and reapply according to the label."
        )

    return " ".join(parts)


def _safety_rules() -> List[Dict]:
    rules = []
    rules.append(
        {
            "tags": "safety:patch_test routine:any",
            "text": (
                "Before using a new cosmetic product on your face, patch test on a small "
                "area of skin and wait at least 24 hours to see how your skin responds."
            ),
        }
    )
    rules.append(
        {
            "tags": "safety:introduce_slowly routine:any",
            "text": (
                "Introduce only one new cosmetic product at a time so you can clearly see "
                "how your skin reacts and avoid overwhelming your routine."
            ),
        }
    )
    rules.append(
        {
            "tags": "safety:avoid_over_exfoliating routine:any",
            "text": (
                "Avoid exfoliating too often. If your skin feels tight, itchy, or looks "
                "very shiny and sensitive, reduce exfoliating products and focus on simple hydration."
            ),
        }
    )
    rules.append(
        {
            "tags": "safety:non_medical routine:any",
            "text": (
                "These suggestions are cosmetic and educational only, and are not a "
                "substitute for professional medical advice, diagnosis, or treatment."
            ),
        }
    )
    return rules


def generate_rules() -> List[Dict]:
    """Generate a list of ~120 synthetic skincare rules."""
    rules: List[Dict] = []

    # Systematically create rules across skin types, concerns, and routines.
    for skin in SKIN_TYPES:
        for concern in CONCERNS:
            for routine in ("am", "pm"):
                tags = f"skin_type:{skin} concern:{concern} routine:{routine}"
                text = _base_rule_text(skin, concern, routine)
                rules.append({"tags": tags, "text": text})

    # Add a few generic routine-structure rules
    generic_rules = [
        {
            "tags": "skin_type:any concern:any routine:am_step_order",
            "text": "A gentle morning routine can follow this order: cleanse, optional mist, serum, moisturizer, and sunscreen.",
        },
        {
            "tags": "skin_type:any concern:any routine:pm_step_order",
            "text": "A simple evening routine can follow this order: cleanse, optional hydrating toner or essence, serum, moisturizer.",
        },
        {
            "tags": "skin_type:any concern:any routine:consistency",
            "text": "Consistency with a comfortable routine usually matters more than frequently swapping products.",
        },
    ]
    rules.extend(generic_rules)

    # Safety and meta rules
    rules.extend(_safety_rules())

    # Attach stable IDs
    with_ids: List[Dict] = []
    for idx, rule in enumerate(rules, start=1):
        with_ids.append(
            {
                "id": f"R{idx:03d}",
                "tags": rule["tags"],
                "text": rule["text"],
            }
        )

    return with_ids


def main() -> None:
    """Write rules.json into the knowledge_base directory."""
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, "rules.json")
    rules = generate_rules()

    os.makedirs(here, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2, ensure_ascii=False)

    print(f"[generate_rules] Wrote {len(rules)} rules to {out_path}")


if __name__ == "__main__":
    main()

