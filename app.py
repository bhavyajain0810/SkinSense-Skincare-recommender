import warnings
warnings.filterwarnings("ignore")
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

load_dotenv()

# ---------------- UI HELPERS ---------------- #

def load_css():
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def hero_section():
    st.markdown("""
    <div style='text-align:center; padding:40px;'>
        <h1 style='color:#ff4b6e;'>✨ SkinSense AI</h1>
        <p style='font-size:18px; color:gray;'>
            Your personal AI skincare assistant
        </p>
    </div>
    """, unsafe_allow_html=True)


def card_start():
    st.markdown('<div class="card">', unsafe_allow_html=True)


def card_end():
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------- CACHE ---------------- #

@st.cache_resource(show_spinner=False)
def _get_chroma_collection():
    try:
        return get_collection()
    except Exception as exc:
        st.error(f"Vector DB error: {exc}")
        return None


# ---------------- RECOMMENDER PAGE ---------------- #

def page_recommender():
    card_start()

    st.subheader("🧴 Personalized Routine Builder")

    st.image("assets/images/hero.jpg", use_container_width=True)

    st.markdown("### 🧬 Tell us about your skin")

    db.init_db()
    collection = _get_chroma_collection()

    uploaded_image = st.file_uploader("Upload face image (optional)", type=["jpg", "png", "jpeg"])

    detected = None
    if uploaded_image:
        detected = detect_from_image(uploaded_image.getvalue())
        if detected:
            st.info(f"Detected: {detected}")

    skin_type = st.selectbox("Skin Type", ["oily", "dry", "combination", "sensitive", "normal"])
    concerns = st.multiselect("Concerns", [
        "acne", "pigmentation", "dullness", "dryness",
        "redness", "texture", "fine_lines", "sun_protection"
    ])

    notes = st.text_area("Extra notes")

    generate = st.button("✨ Generate My Routine", use_container_width=True)

    card_end()

    if generate:
        if not collection:
            st.error("Database not ready")
            return

        attrs = {"skin_type": skin_type, "concerns": concerns, "notes": notes}
        query = make_query(attrs)

        with st.spinner("🔍 Analyzing your skin..."):
            retrieved = retrieve_rules(collection, query=query, k=8)

        if not retrieved:
            st.error("No results found")
            return

        rule_ids = [r["id"] for r in retrieved if r.get("id")]
        prompt = build_prompt(attrs, retrieved)

        try:
            with st.spinner("🤖 Generating routine..."):
                response_md = call_llm(prompt)
        except Exception as exc:
            st.error(f"LLM ERROR: {exc}")
            import traceback
            st.code(traceback.format_exc())
            response_md = fallback_answer(rule_ids)

        db.insert_interaction(attrs, rule_ids, response_md)

        # Output Card
        card_start()

        st.success("✨ Routine generated successfully!")

        st.markdown("### 🤖 Your AI Routine")

        st.markdown(f"""
        <div style='background:#f9f9f9; padding:20px; border-radius:15px;'>
        {response_md}
        </div>
        """, unsafe_allow_html=True)

        card_end()


# ---------------- DASHBOARD ---------------- #

def page_dashboard():
    st.subheader("📊 Usage Dashboard")

    interactions = db.fetch_all_interactions()
    if not interactions:
        st.info("No data yet")
        return

    df = pd.DataFrame([{
        "skin_type": i["attributes"].get("skin_type", "unknown"),
        "feedback": i.get("feedback", "none")
    } for i in interactions])

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(df["skin_type"].value_counts())
        fig.update_layout(template="plotly_white")
        card_start()
        st.plotly_chart(fig, use_container_width=True)
        card_end()

    with col2:
        fig2 = px.bar(df["feedback"].value_counts())
        fig2.update_layout(template="plotly_white")
        card_start()
        st.plotly_chart(fig2, use_container_width=True)
        card_end()


# ---------------- MAIN ---------------- #

def main():
    st.set_page_config(page_title="SkinSense AI", layout="wide")

    load_css()
    hero_section()

    page = st.sidebar.radio("Navigation", ["🏠 Recommender", "📊 Dashboard"])

    if page == "🏠 Recommender":
        page_recommender()
    else:
        page_dashboard()


if __name__ == "__main__":
    main()