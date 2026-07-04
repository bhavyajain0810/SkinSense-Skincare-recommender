# Sample inputs and outputs

These examples are recruiter-facing demonstrations of the product contract. Exact retrieved distances can vary with model/library versions.

## Oily skin with acne concerns

Input:

```json
{
  "skin_type": "oily",
  "concerns": ["acne"],
  "notes": "I prefer a short beginner routine."
}
```

Expected behavior:

- Retrieve cards tagged for oily skin, acne, routine order, or cosmetic safety.
- Return separate AM and PM routines.
- Favor lightweight textures, gentle cleansing, moisturizer, and sunscreen.
- Include only retrieved `Rxxx` IDs in the citations expander.

Representative output shape:

```markdown
Here is a gentle routine focused on comfortable, lightweight steps.

## AM Routine
- Cleanse gently.
- Use a comfortable lightweight moisturizer.
- Finish with cosmetic sunscreen according to its label.

## PM Routine
- Remove sunscreen and daily buildup with a mild cleanser.
- Add a light hydrating layer.
- Finish with a comfortable moisturizer.

## Extra Tips
- Introduce one new product at a time.
- Patch test new cosmetic products.

## Why these suggestions?
The retrieved cards emphasize lightweight textures and a consistent routine.

## Citations
Used: R001, R002, R084
```

## Dry, sensitive-feeling skin

Input:

```json
{
  "skin_type": "dry",
  "concerns": ["dryness", "redness"],
  "notes": "Strong products often feel uncomfortable."
}
```

Expected behavior:

- Prefer soft, cushioning textures and simple fragrance-free formulas.
- Avoid aggressive exfoliation or treatment claims.
- Show a professional-care boundary if notes describe severe or worsening symptoms.

## No language service available

Input: any valid profile while the FastAPI service is stopped.

Expected behavior:

- Retrieval continues locally.
- SkinSense displays the deterministic fallback with the required sections.
- Retrieved source cards and their distances remain available.
- The interface identifies fallback mode without showing a traceback.

## Invalid input

Input:

```json
{
  "skin_type": "unknown",
  "concerns": ["acne"]
}
```

Expected behavior: generation is blocked with an unsupported skin-type message. No retrieval, API call, or database insert occurs.
