import streamlit as st

from src.i18n import t, detect_language

# ── Language detection ──
if "lang" not in st.session_state:
    qp_lang = st.query_params.get("lang")
    if qp_lang and qp_lang in ("pt_BR", "en", "es"):
        st.session_state["lang"] = qp_lang
    else:
        st.session_state["lang"] = "en"
        detect_language()

lang = st.session_state["lang"]

st.set_page_config(
    page_title="LLM Session Summarizer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──
st.markdown(
    """
<style>
.stApp { background-color: #0e1117; color: #e0e0e0; }
h1, h2, h3, p, li { color: #e0e0e0 !important; }
code { background-color: #1e1e2e !important; color: #cdd6f4 !important; }
a { color: #4285f4 !important; }
.stButton > button[kind="primary"] { background-color: #22c55e !important; border-color: #16a34a !important; }
.stButton > button[kind="primary"]:hover { background-color: #16a34a !important; }
[data-testid="stSidebarCollapseButton"] { top: 50vh; transform: translateY(-50%); }
</style>
""",
    unsafe_allow_html=True,
)

col = st.columns([1, 3, 1])[1]

with col:
    # Lang selector — top-right, compact
    _, lang_col = st.columns([5, 1])
    with lang_col:
        lang_labels = {"pt_BR": "PT", "en": "EN", "es": "ES"}
        selected = st.selectbox(
            "Idioma",
            options=list(lang_labels.keys()),
            format_func=lambda l: lang_labels[l],
            index=list(lang_labels.keys()).index(lang),
            label_visibility="collapsed",
            key="lang_selector",
        )
        if selected != lang:
            st.session_state["lang"] = selected
            st.query_params["lang"] = selected
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.title("🔬 LLM Session Summarizer")

    st.markdown(f"### 🚀 {t('hero.tagline', lang)}")

    st.markdown(t("hero.subtitle", lang))

    st.caption(
        "⚡ 4 provedores  ·  6 secoes por resumo  ·  Streaming token a token  ·  "
        "📥 Download .md  ·  🤖 Prompt de continuidade  ·  ⚡ TOON"
    )

    if st.button(t("hero.cta", lang), type="primary", use_container_width=True):
        st.switch_page("pages/1_Summarizer.py")

    st.divider()

    st.markdown(f"### 🚀 {t('howto.title', lang)}")

    if st.button(t("howto.step1", lang), type="primary", use_container_width=True, key="step1"):
        st.switch_page("pages/1_Summarizer.py")
    st.markdown(t("howto.steps", lang))

    st.divider()

    st.markdown(f"### 🧩 {t('providers.title', lang)}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**🦙 {t('providers.ollama.name', lang)}**\n{t('providers.ollama.desc', lang)}")
        st.markdown(f"**🔷 {t('providers.openai.name', lang)}**\n{t('providers.openai.desc', lang)}")
    with col2:
        st.markdown(f"**🤖 {t('providers.gemini.name', lang)}**\n{t('providers.gemini.desc', lang)}")
        st.markdown(f"**🧠 {t('providers.anthropic.name', lang)}**\n{t('providers.anthropic.desc', lang)}")

    st.divider()

    st.markdown(f"### ⚡ {t('toon.title', lang)}")
    st.markdown(t("toon.desc", lang))

    st.divider()

    st.caption(t("footer", lang))
