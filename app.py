import json
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from rag.retrieve import get_collection, retrieve_rules
from utils import db
from utils.llm_client import LLMConfigurationError, call_llm, fallback_answer
from utils.prompt_templates import build_prompt, make_query
from utils.vision_attributes import detect_from_image


load_dotenv()  # Load .env for local runs; Docker uses env vars directly.


@st.cache_resource(show_spinner=False)
def _get_chroma_collection():
    """Cache the Chroma collection across reruns."""
    try:
        collection = get_collection()
        return collection
    except Exception as exc:
        st.error(
            f"Could not connect to the vector database. "
            f"Please run `python rag/build_index.py` or `python scripts/bootstrap.py`.\n\nDetails: {exc}"
        )
        return None


def _prepare_dashboard_data():
    interactions = db.fetch_all_interactions()
    if not interactions:
        return None, None, None, None, None

    # Build a flat DataFrame for plotting.
    rows = []
    for item in interactions:
        attrs = item.get("attributes") or {}
        skin_type = attrs.get("skin_type") or "unknown"
        concerns = attrs.get("concerns") or []
        if isinstance(concerns, str):
            concerns_list = [c.strip() for c in concerns.split(",") if c.strip()]
        else:
            concerns_list = concerns

        rows.append(
            {
                "id": item["id"],
                "ts": item["ts"],
                "skin_type": skin_type,
                "concerns": ",".join(concerns_list),
                "retrieved_rule_ids": item.get("retrieved_rule_ids", ""),
                "feedback": item.get("feedback") or "none",
            }
        )

    df = pd.DataFrame(rows)

    # Skin type frequency
    skin_counts = df["skin_type"].value_counts().reset_index()
    skin_counts.columns = ["skin_type", "count"]

    # Concerns frequency (top 15, sorted descending)
    all_concerns = []
    for c_str in df["concerns"]:
        for c in (c_str or "").split(","):
            c = c.strip()
            if c:
                all_concerns.append(c)
    if all_concerns:
        concerns_df = (
            pd.Series(all_concerns)
            .value_counts()
            .head(15)
            .reset_index()
        )
        concerns_df.columns = ["concern", "count"]
    else:
        concerns_df = pd.DataFrame(columns=["concern", "count"])

    # Feedback ratio
    feedback_df = df["feedback"].value_counts().reset_index()
    # After value_counts().reset_index(), columns are typically [feedback, count]
    # Ensure they are named consistently for plotting.
    feedback_df.columns = ["feedback", "count"]

    # Recent interactions
    recent = db.fetch_recent_interactions(limit=25)
    recent_rows = []
    for item in recent:
        attrs = item.get("attributes") or {}
        skin_type = attrs.get("skin_type") or "unknown"
        concerns = attrs.get("concerns") or []
        if isinstance(concerns, str):
            concerns_list = [c.strip() for c in concerns.split(",") if c.strip()]
        else:
            concerns_list = concerns
        recent_rows.append(
            {
                "id": item["id"],
                "ts": item["ts"],
                "skin_type": skin_type,
                "concerns": ",".join(concerns_list),
                "retrieved_rule_ids": item.get("retrieved_rule_ids", ""),
                "feedback": item.get("feedback") or "",
            }
        )
    recent_df = pd.DataFrame(recent_rows)

    return df, skin_counts, concerns_df, feedback_df, recent_df


def page_recommender():
    st.subheader("Personalized Skincare Recommender")
    st.write(
        "This tool suggests gentle, cosmetic-only routines using a local knowledge base. "
        "It does not provide medical advice."
    )
    with st.expander("About SkinSense / Safety Disclaimer"):
     st.markdown(
        """
        **SkinSense is a cosmetic-only educational demo.**

        - It does **not** diagnose skin conditions or provide medical treatment advice.
        - It suggests gentle routines based on a local skincare rules knowledge base.
        - If you have severe irritation, pain, swelling, infection, or worsening symptoms, seek help from a qualified professional.

        **How the recommendation works**

        1. You enter your skin type, concerns, and optional notes.
        2. SkinSense retrieves the most relevant rule cards from its local knowledge base using RAG (retrieval-augmented generation).
        3. The app sends those retrieved rules to the language model and asks it to generate a routine grounded only in those rules.
        4. If the live LLM is unavailable, the app falls back to a safe built-in template.

        The goal is to keep recommendations **gentle, simple, transparent, and non-medical**.
        """
    )
    
    db.init_db()
    collection = _get_chroma_collection()

    # Optional image upload
    uploaded_image = st.file_uploader(
        "Optional: upload a clear face photo (jpg/png). "
        "If a vision service is configured, it may help pre-fill your skin type and concerns.",
        type=["jpg", "jpeg", "png"],
    )

    detected = None
    if uploaded_image is not None:
        try:
            detected = detect_from_image(uploaded_image.getvalue())
            if detected:
                st.info(
                    "Detected from image (cosmetic estimate only): "
                    + ", ".join(
                        [
                            f"skin type: {detected.get('skin_type', 'n/a')}",
                            f"concerns: {', '.join(detected.get('concerns', [])) or 'n/a'}",
                        ]
                    )
                )
        except Exception as exc:  # pragma: no cover - fully defensive
            st.warning(f"Could not process image for attributes: {exc}")

    skin_options = ["oily", "dry", "combination", "sensitive", "normal"]
    concern_options = [
        "acne",
        "pigmentation",
        "dullness",
        "dryness",
        "redness",
        "texture",
        "fine_lines",
        "sun_protection",
    ]

    # Defaults from detection if available; Streamlit keys keep user edits.
    default_skin_index = 0
    if detected and detected.get("skin_type") in skin_options:
        default_skin_index = skin_options.index(detected["skin_type"])

    default_concerns = []
    if detected and detected.get("concerns"):
        default_concerns = [
            c for c in detected["concerns"] if c in concern_options
        ]

    default_notes = ""
    if detected and detected.get("notes"):
        default_notes = detected["notes"]

    col1, col2 = st.columns(2)
    with col1:
        skin_type = st.selectbox(
            "Skin type",
            options=skin_options,
            index=default_skin_index,
            key="skin_type_input",
            help="Choose the skin type that best matches how your skin behaves most of the time.",
        )
    with col2:
        concerns = st.multiselect(
            "Top concerns (choose up to 3)",
            options=concern_options,
            default=default_concerns,
            key="concerns_input",
            help="Select up to 3 main cosmetic concerns you want the routine to focus on.",
        )

    notes = st.text_area(
        "Extra notes (optional, e.g. 'prefers very simple routine')",
        value=default_notes,
        key="notes_input",
    )

    k = st.slider(
        "Number of rule cards to use (k)",
        min_value=5,
        max_value=15,
        value=8,
        step=1,
        help="Controls how many retrieved rule cards are used to build the recommendation.",
    )

    if len(concerns) > 3:
        st.warning("Please select at most 3 concerns.")

    if "last_interaction_id" not in st.session_state:
        st.session_state["last_interaction_id"] = None
    if "last_rule_ids" not in st.session_state:
        st.session_state["last_rule_ids"] = []

    if st.button("Generate routine", type="primary", disabled=collection is None):
        if collection is None:
            st.error("Vector database not ready. See message above.")
            return

        if len(concerns) > 3:
            st.error("Too many concerns selected. Please choose at most 3.")
            return

        attrs = {
            "skin_type": skin_type,
            "concerns": concerns,
            "notes": notes,
        }

        query = make_query(attrs)
        with st.spinner("Retrieving relevant skincare rules..."):
            retrieved = retrieve_rules(collection, query=query, k=k)

        if not retrieved:
            st.error(
                "No rules could be retrieved. Try broadening your concerns or check that "
                "the knowledge base and index are built."
            )
            return
        
        with st.expander("Retrieved rules details"):
            details_rows = []
            for rule in retrieved:
                meta = rule.get("metadata") or {}
                distance = rule.get("distance")
                details_rows.append(
                    {
                        "id": rule.get("id", ""),
                        "tags": meta.get("tags", "") if isinstance(meta, dict) else "",
                        "distance": round(float(distance), 4) if distance is not None else None,
                    }
                )
            st.table(details_rows)        

        rule_ids = [r["id"] for r in retrieved if r.get("id")]

        prompt = build_prompt(attrs, retrieved)

        try:
            with st.spinner("Calling LLM to generate a routine..."):
                response_md = call_llm(prompt)
        except LLMConfigurationError:
            st.warning(
                "LLM endpoint is not configured. Using a built-in fallback template instead."
            )
            response_md = fallback_answer(rule_ids)
        except Exception as exc:
            st.error(
                "The LLM request failed. Showing a safe fallback routine instead. "
                f"Details: {exc}"
            )
            response_md = fallback_answer(rule_ids)

        try:
            interaction_id = db.insert_interaction(
                attributes=attrs,
                retrieved_rule_ids=rule_ids,
                response_md=response_md,
            )
        except Exception as exc:
            st.error(f"Could not save interaction to SQLite: {exc}")
            interaction_id = None

        export_payload = {
                  "interaction_id": interaction_id,
                  "attributes": attrs,
                  "retrieved_rule_ids": rule_ids,
                  "response_md": response_md,
            }

        st.download_button(
            label="Export last interaction (JSON)",
            data=json.dumps(export_payload, indent=2, ensure_ascii=False),
            file_name="skinsense_last_interaction.json",
            mime="application/json",
        )
      
        st.session_state["last_interaction_id"] = interaction_id
        st.session_state["last_rule_ids"] = rule_ids

        st.markdown(response_md)
        st.caption(
            "Retrieved rule IDs: "
            + (", ".join(rule_ids) if rule_ids else "N/A")
        )

    # Feedback section
    last_id = st.session_state.get("last_interaction_id")
    if last_id:
        st.markdown("---")
        st.write("Was this recommendation helpful?")
        fb_col1, fb_col2 = st.columns(2)

        with fb_col1:
            if st.button("👍 Helpful"):
                try:
                    db.update_feedback(int(last_id), "helpful")
                    st.success("Thanks for your feedback!")
                except Exception as exc:
                    st.error(f"Could not save feedback: {exc}")
        with fb_col2:
            if st.button("👎 Not helpful"):
                try:
                    db.update_feedback(int(last_id), "not_helpful")
                    st.info("Feedback noted — thank you.")
                except Exception as exc:
                    st.error(f"Could not save feedback: {exc}")


def page_dashboard():
    st.subheader("Usage Dashboard")
    st.write(
        "Simple visualizations of how the recommender is being used. "
        "All data is stored locally in a SQLite file under the `logs` directory."
    )

    result = _prepare_dashboard_data()
    if result is None or result[0] is None:
        st.info("No interactions have been logged yet.")
        st.markdown(
            """
            ### Next steps
            To populate the dashboard:

            1. Go to the **Recommender** tab.
            2. Choose a skin type and at least one concern.
            3. Click **Generate routine**.
            4. Return to the **Dashboard** tab to view charts and recent interactions.

            Once a few recommendations are generated, this dashboard will show:
            - skin type frequency
            - top concerns
            - feedback breakdown
            - recent interaction history (last 25)
            """
        )
        return

    _, skin_counts, concerns_df, feedback_df, recent_df = result

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Skin type frequency")
        if skin_counts is not None and not skin_counts.empty:
            fig_skin = px.bar(
                skin_counts,
                x="skin_type",
                y="count",
                title="Skin types seen",
            )
            st.plotly_chart(fig_skin, use_container_width=True)
        else:
            st.write("No skin type data yet.")

    with col2:
        st.markdown("#### Top concerns frequency")
        if concerns_df is not None and not concerns_df.empty:
            fig_concerns = px.bar(
                concerns_df,
                x="concern",
                y="count",
                title="Most common concerns",
            )
            st.plotly_chart(fig_concerns, use_container_width=True)
        else:
            st.write("No concern data yet.")

    st.markdown("#### Feedback helpful vs not helpful")
    if feedback_df is not None and not feedback_df.empty:
        fig_fb = px.bar(
            feedback_df,
            x="feedback",
            y="count",
            title="Feedback distribution",
        )
        st.plotly_chart(fig_fb, use_container_width=True)
    else:
        st.write("No feedback recorded yet.")

        st.markdown("#### Recent interactions (last 25)")
    if recent_df is not None and not recent_df.empty:
        filter_col1, filter_col2 = st.columns(2)

        skin_type_options = ["All"] + sorted(
            [str(x) for x in recent_df["skin_type"].dropna().unique().tolist()]
        )
        feedback_options = ["All"] + sorted(
            [str(x) for x in recent_df["feedback"].dropna().unique().tolist()]
        )

        with filter_col1:
            selected_skin_type = st.selectbox(
                "Filter by skin type",
                options=skin_type_options,
                key="dashboard_skin_type_filter",
            )

        with filter_col2:
            selected_feedback = st.selectbox(
                "Filter by feedback",
                options=feedback_options,
                key="dashboard_feedback_filter",
            )

        filtered_recent_df = recent_df.copy()

        if selected_skin_type != "All":
            filtered_recent_df = filtered_recent_df[
                filtered_recent_df["skin_type"] == selected_skin_type
            ]

        if selected_feedback != "All":
            filtered_recent_df = filtered_recent_df[
                filtered_recent_df["feedback"] == selected_feedback
            ]

        st.dataframe(filtered_recent_df, use_container_width=True)
    else:
        st.write("No recent interactions yet.")


def main():
    st.set_page_config(
        page_title="SkinSense: Skincare Recommender",
        layout="wide",
    )

    st.title("SkinSense: Skincare Recommender (RAG + LLM)")

    tab1, tab2 = st.tabs(["Recommender", "Dashboard"])

    with tab1:
        page_recommender()
    with tab2:
        page_dashboard()


if __name__ == "__main__":
    main()

