"""SkinSense Streamlit application."""

import hashlib
import html
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from rag.retrieve import (
    IndexNotReadyError,
    RetrievalError,
    get_collection,
    retrieve_rules,
)
from utils import db
from utils.llm_client import (
    LLMConfigurationError,
    LLMResponseError,
    LLMServiceError,
    call_llm,
    check_llm_health,
    fallback_answer,
)
from utils.prompt_templates import build_prompt, make_query
from utils.validation import (
    CONCERNS,
    SKIN_TYPES,
    InputValidationError,
    validate_skin_profile,
)
from utils.vision_attributes import detect_from_image


PROJECT_ROOT = Path(__file__).resolve().parent
SECTION_ORDER = (
    "AM Routine",
    "PM Routine",
    "Extra Tips",
    "Why these suggestions?",
    "Citations",
)
PASTEL_COLORS = ["#9BAF9A", "#D7A9B5", "#AFA6C9", "#E7B59B", "#C8A77B"]

load_dotenv()


def load_css() -> None:
    css = (PROJECT_ROOT / "assets" / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def hero_section() -> None:
    st.markdown(
        """
        <section class="brand-hero">
          <div>
            <p class="eyebrow">PERSONAL ROUTINE STUDIO</p>
            <h1>Skincare, made easier to understand.</h1>
            <p class="hero-copy">
              Build a calm, cosmetic-only routine grounded in a curated knowledge base.
              No miracle claims—just clear steps and traceable guidance.
            </p>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def _get_chroma_collection():
    return get_collection()


@st.cache_data(ttl=10, show_spinner=False)
def _cached_health() -> Dict[str, Any]:
    return check_llm_health()


def _load_collection():
    try:
        return _get_chroma_collection()
    except IndexNotReadyError:
        return None
    except Exception:
        return None


def _initialize_state() -> None:
    defaults = {
        "skin_type_input": None,
        "concerns_input": [],
        "notes_input": "",
        "last_result": None,
        "feedback_saved": None,
        "vision_digest": None,
        "vision_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _handle_uploaded_image(uploaded_image) -> None:
    if uploaded_image is None:
        return
    image_bytes = uploaded_image.getvalue()
    digest = hashlib.sha256(image_bytes).hexdigest()
    if digest == st.session_state.vision_digest:
        return

    st.session_state.vision_digest = digest
    detected = detect_from_image(image_bytes)
    st.session_state.vision_result = detected
    if not detected:
        return
    if detected.get("skin_type") in SKIN_TYPES:
        st.session_state.skin_type_input = detected["skin_type"]
    if detected.get("concerns"):
        st.session_state.concerns_input = detected["concerns"]
    if detected.get("notes"):
        st.session_state.notes_input = detected["notes"]


def _parse_response(markdown: str) -> Dict[str, str]:
    pattern = re.compile(
        r"^##\s+(AM Routine|PM Routine|Extra Tips|Why these suggestions\?|Citations)\s*$",
        flags=re.MULTILINE,
    )
    matches = list(pattern.finditer(markdown))
    sections: Dict[str, str] = {}
    if matches and matches[0].start() > 0:
        sections["Overview"] = markdown[: matches[0].start()].strip()
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        sections[match.group(1)] = markdown[match.end() : end].strip()
    return sections


def _sidebar(collection) -> int:
    with st.sidebar:
        st.markdown('<p class="sidebar-brand">SkinSense</p>', unsafe_allow_html=True)
        st.caption("Local settings and service status")
        st.divider()

        k = st.slider(
            "Retrieved knowledge cards",
            min_value=4,
            max_value=12,
            value=8,
            help="More cards provide broader context; fewer cards keep the prompt focused.",
        )

        st.markdown("#### System status")
        index_ready = collection is not None
        st.markdown(
            f'<div class="status-row"><span>Knowledge index</span>'
            f'<span class="status-dot {"ready" if index_ready else "offline"}">'
            f'{"Ready" if index_ready else "Missing"}</span></div>',
            unsafe_allow_html=True,
        )

        if st.button("Check language service", use_container_width=True):
            _cached_health.clear()
        health = _cached_health()
        backend_label = html.escape(health["backend"].title())
        st.markdown(
            f'<div class="status-row"><span>Language service</span>'
            f'<span class="status-dot {"ready" if health["available"] else "offline"}">'
            f'{backend_label}</span></div>',
            unsafe_allow_html=True,
        )
        st.caption(health["base_url"])

        st.divider()
        st.caption(
            "SkinSense provides cosmetic and educational guidance only. "
            "It does not diagnose or treat skin conditions."
        )
    return k


def _show_profile_form() -> bool:
    st.markdown(
        """
        <div class="section-heading">
          <p class="section-kicker">YOUR PROFILE</p>
          <h2>Let’s build a routine that fits</h2>
          <p>Choose what feels closest today. You can adjust it whenever you like.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 0.95], gap="medium")
    with left:
        with st.container(border=True):
            st.markdown("#### Skin profile")
            st.selectbox(
                "Skin type",
                options=list(SKIN_TYPES),
                index=None,
                placeholder="Select your skin type",
                format_func=lambda value: value.replace("_", " ").title(),
                key="skin_type_input",
            )
            st.multiselect(
                "Main concerns",
                options=list(CONCERNS),
                format_func=lambda value: value.replace("_", " ").title(),
                placeholder="Choose one or more concerns",
                key="concerns_input",
            )
            st.text_area(
                "Anything else we should consider?",
                placeholder="For example: I prefer a very simple routine.",
                max_chars=1_000,
                key="notes_input",
            )

    with right:
        with st.container(border=True):
            st.markdown("#### Optional image input")
            st.caption(
                "If you have configured a local vision service, an image can suggest "
                "editable profile defaults. The recommender still works without one."
            )
            uploaded_image = st.file_uploader(
                "Upload a face image",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
            )
            _handle_uploaded_image(uploaded_image)
            detected = st.session_state.vision_result
            if detected:
                labels = []
                if detected.get("skin_type"):
                    labels.append(detected["skin_type"].title())
                labels.extend(
                    concern.replace("_", " ").title()
                    for concern in detected.get("concerns", [])
                )
                st.success("Suggested profile: " + ", ".join(labels))
            elif uploaded_image is not None:
                st.info("No local vision result was available. Manual selections remain active.")

            st.markdown(
                """
                <div class="ingredient-note">
                  <p class="ingredient-label">HOW RECOMMENDATIONS ARE MADE</p>
                  <p>Your profile becomes a search query. Relevant cosmetic rules are
                  retrieved, cited, and used to shape the final routine.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    return st.button(
        "Build my routine",
        type="primary",
        use_container_width=True,
    )


def _generate_routine(collection, k: int) -> None:
    st.session_state.last_result = None
    st.session_state.feedback_saved = None

    try:
        attributes = validate_skin_profile(
            {
                "skin_type": st.session_state.skin_type_input,
                "concerns": st.session_state.concerns_input,
                "notes": st.session_state.notes_input,
            }
        )
    except InputValidationError as exc:
        st.warning(str(exc))
        return

    if collection is None:
        st.error(
            "The local knowledge index is not ready. Run "
            "`python scripts/bootstrap.py`, then refresh this page."
        )
        return

    try:
        with st.spinner("Finding the most relevant skincare guidance…"):
            retrieved = retrieve_rules(collection, query=make_query(attributes), k=k)
    except (InputValidationError, IndexNotReadyError, RetrievalError) as exc:
        st.error(str(exc))
        return

    if not retrieved:
        st.warning(
            "No matching knowledge cards were found. Try a broader concern or rebuild "
            "the local index."
        )
        return

    rule_ids = [rule["id"] for rule in retrieved if rule.get("id")]
    used_fallback = False
    service_note = ""
    try:
        with st.spinner("Composing your routine…"):
            response_md = call_llm(build_prompt(attributes, retrieved))
    except (LLMConfigurationError, LLMServiceError, LLMResponseError, ValueError) as exc:
        response_md = fallback_answer(rule_ids)
        used_fallback = True
        service_note = str(exc)
    except Exception:
        response_md = fallback_answer(rule_ids)
        used_fallback = True
        service_note = "The language service could not complete the request."

    interaction_id = None
    try:
        interaction_id = db.insert_interaction(attributes, rule_ids, response_md)
    except (db.DatabaseError, ValueError):
        service_note = (
            f"{service_note} " if service_note else ""
        ) + "The routine could not be added to local history."

    st.session_state.last_result = {
        "attributes": attributes,
        "response_md": response_md,
        "retrieved": retrieved,
        "interaction_id": interaction_id,
        "used_fallback": used_fallback,
        "service_note": service_note.strip(),
    }


def _render_result() -> None:
    result = st.session_state.last_result
    if not result:
        return

    st.markdown("---")
    st.markdown('<p class="section-kicker">YOUR ROUTINE</p>', unsafe_allow_html=True)
    st.markdown("## A simple plan for morning and evening")
    if result["used_fallback"]:
        st.info(
            "The local fallback prepared this routine. Your retrieved knowledge cards "
            "and citations are still shown below."
        )
    if result["service_note"] and "history" in result["service_note"].lower():
        st.warning(result["service_note"])

    sections = _parse_response(result["response_md"])
    if sections.get("Overview"):
        st.markdown(sections["Overview"])

    first, second = st.columns(2, gap="large")
    for column, section_name in zip((first, second), ("AM Routine", "PM Routine")):
        with column:
            with st.container(border=True):
                st.markdown(f"#### {section_name}")
                st.markdown(sections.get(section_name, "No steps were returned."))

    with st.container(border=True):
        st.markdown("#### Extra Tips")
        st.markdown(sections.get("Extra Tips", "Keep your routine gentle and consistent."))

    with st.expander("Why these suggestions and source cards"):
        st.markdown(sections.get("Why these suggestions?", ""))
        st.markdown("##### Retrieved knowledge cards")
        for rule in result["retrieved"]:
            st.caption(f'{rule["id"]} · distance {rule["distance"]:.4f}')
            st.write(rule["document"])
        st.markdown("##### Citations")
        st.markdown(sections.get("Citations", ""))

    interaction_id = result["interaction_id"]
    if interaction_id is not None:
        st.markdown("#### Was this routine useful?")
        helpful, not_helpful, spacer = st.columns([1, 1, 3])
        with helpful:
            helpful_clicked = st.button(
                "Helpful",
                use_container_width=True,
                key=f"helpful_{interaction_id}",
            )
        with not_helpful:
            not_helpful_clicked = st.button(
                "Not helpful",
                use_container_width=True,
                key=f"not_helpful_{interaction_id}",
            )
        if helpful_clicked or not_helpful_clicked:
            feedback = "helpful" if helpful_clicked else "not_helpful"
            try:
                db.update_feedback(interaction_id, feedback)
                st.session_state.feedback_saved = feedback
            except (db.DatabaseError, db.InteractionNotFoundError, ValueError):
                st.error("Feedback could not be saved to local history.")
        if st.session_state.feedback_saved:
            st.success("Feedback saved. Thank you.")


def page_recommender(collection, k: int) -> None:
    if _show_profile_form():
        _generate_routine(collection, k)
    _render_result()


def _style_chart(fig, title: str) -> None:
    fig.update_layout(
        title=title,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#443E3A", "family": "Arial"},
        title_font={"size": 17},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, title=None)
    fig.update_yaxes(gridcolor="#EDE5DF", title=None)


def page_dashboard() -> None:
    st.markdown('<p class="section-kicker">LOCAL INSIGHTS</p>', unsafe_allow_html=True)
    st.markdown("## Routine activity at a glance")
    st.caption("A private view of interactions stored in your local SQLite database.")

    try:
        interactions = db.fetch_all_interactions()
    except db.DatabaseError:
        st.error("Local dashboard data is unavailable right now.")
        return

    if not interactions:
        with st.container(border=True):
            st.info("No routines have been logged yet. Build one to populate this view.")
        return

    rows: List[Dict[str, Any]] = []
    concern_rows: List[str] = []
    for item in interactions:
        attrs = item.get("attributes") or {}
        concerns = attrs.get("concerns") or []
        concern_rows.extend(concerns)
        rows.append(
            {
                "id": item["id"],
                "timestamp": item["ts"],
                "skin_type": attrs.get("skin_type", "unknown"),
                "concerns": ", ".join(concerns) or "none",
                "rule_ids": item.get("retrieved_rule_ids", ""),
                "feedback": item.get("feedback") or "none",
            }
        )
    frame = pd.DataFrame(rows)
    helpful_count = int((frame["feedback"] == "helpful").sum())
    rated_count = int(frame["feedback"].isin(["helpful", "not_helpful"]).sum())
    helpful_rate = round((helpful_count / rated_count) * 100) if rated_count else 0
    top_skin_type = frame["skin_type"].mode().iat[0].replace("_", " ").title()

    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.metric("Routines created", len(frame))
    metric_two.metric("Most common profile", top_skin_type)
    metric_three.metric("Helpful rating", f"{helpful_rate}%" if rated_count else "No ratings")

    skin_counts = frame["skin_type"].value_counts().rename_axis("skin_type").reset_index(name="count")
    skin_fig = px.bar(
        skin_counts,
        x="skin_type",
        y="count",
        color="skin_type",
        color_discrete_sequence=PASTEL_COLORS,
    )
    _style_chart(skin_fig, "Skin profiles")

    feedback_counts = frame["feedback"].value_counts().rename_axis("feedback").reset_index(name="count")
    feedback_fig = px.bar(
        feedback_counts,
        x="feedback",
        y="count",
        color="feedback",
        color_discrete_sequence=["#9BAF9A", "#D7A9B5", "#D8D0C8"],
    )
    _style_chart(feedback_fig, "Routine feedback")

    chart_one, chart_two = st.columns(2, gap="large")
    with chart_one:
        with st.container(border=True):
            st.plotly_chart(skin_fig, use_container_width=True)
    with chart_two:
        with st.container(border=True):
            st.plotly_chart(feedback_fig, use_container_width=True)

    if concern_rows:
        concern_counts = pd.Series(concern_rows).value_counts().head(8).rename_axis("concern").reset_index(name="count")
        concern_fig = px.bar(
            concern_counts,
            x="count",
            y="concern",
            orientation="h",
            color="concern",
            color_discrete_sequence=PASTEL_COLORS,
        )
        _style_chart(concern_fig, "Most selected concerns")
        with st.container(border=True):
            st.plotly_chart(concern_fig, use_container_width=True)

    with st.expander("Recent local interactions", expanded=False):
        st.dataframe(
            frame.sort_values("id", ascending=False).head(25),
            use_container_width=True,
            hide_index=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="SkinSense",
        page_icon="🧴",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    load_css()
    _initialize_state()
    try:
        db.init_db()
    except db.DatabaseError:
        st.warning("Local history is unavailable; recommendations can still be generated.")

    collection = _load_collection()
    k = _sidebar(collection)
    hero_section()

    page = st.radio(
        "View",
        ["Routine studio", "Insights"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if page == "Routine studio":
        page_recommender(collection, k)
    else:
        page_dashboard()


if __name__ == "__main__":
    main()
