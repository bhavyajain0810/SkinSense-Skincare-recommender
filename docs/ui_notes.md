# UI design notes

## Product direction

SkinSense should feel like a calm skincare product: warm, useful, and restrained. It should not look like an AI control panel or a medical scanner.

## Visual system

- Background: warm off-white and cream.
- Accents: soft blush, muted lavender, sage, and peach.
- Typography: readable system sans-serif with a quiet editorial serif for the hero and metrics.
- Surfaces: rounded native Streamlit containers, subtle borders, and low-contrast shadows.
- Motion: only short button hover feedback; no animated gradients or visual spectacle.

## Content principles

- Lead with the user’s routine, not the implementation.
- Avoid “AI-powered,” “magic,” diagnostic, or result-guarantee language.
- Keep safety boundaries present but not alarming.
- Put backend configuration and status in the sidebar.
- Put retrieval evidence behind an expander so it is available without dominating the routine.
- Keep feedback simple: Helpful / Not helpful.

## Layout

- Editorial hero and compact page switcher.
- Two-column profile area on desktop; stacked on mobile.
- Separate AM and PM result cards, followed by tips.
- Source rationale, rule IDs, and distances in one subtle expander.
- Dashboard uses the same palette, metric cards, focused charts, and a collapsed recent-history table.

## Accessibility

- Maintain readable contrast against pastel surfaces.
- Use text labels in addition to color for service status.
- Prefer native Streamlit controls for keyboard and screen-reader behavior.
- Do not place essential information only in decorative imagery.
