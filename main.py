import streamlit as st

st.set_page_config(
    page_title="LLM Session Summarizer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="auto",
)

# ── CSS ──
st.markdown(
    """
<style>
.stApp { background-color: #0e1117; color: #e0e0e0; }
h1, h2, h3, p, li { color: #e0e0e0 !important; }
code { background-color: #1e1e2e !important; color: #cdd6f4 !important; }
a { color: #4285f4 !important; }
</style>
""",
    unsafe_allow_html=True,
)

col = st.columns([1, 3, 1])[1]

with col:
    st.markdown("<br>", unsafe_allow_html=True)
    st.title("🔬 LLM Session Summarizer")

    st.markdown("### Transforme conversas com IAs em conhecimento estruturado")

    st.page_link("pages/1_Summarizer.py", label="Abrir Summarizer", icon=":material/arrow_forward:")

    st.markdown("""
    Faça upload de arquivos `.json` (Gemini CLI) ou `.md` (OpenCode) e escolha
    entre **4 provedores** — Ollama, Gemini, OpenAI e Anthropic — para gerar
    resumos estruturados em **6 seções** com streaming, downloads em markdown e
    **formato TOON para economia de tokens**.
    """)

    st.divider()

    st.markdown("### 🚀 Como usar")

    st.page_link("pages/1_Summarizer.py", label="1. Abra o Summarizer", icon=":material/arrow_forward:")
    st.markdown("""
    2. Escolha o provedor: **Ollama**, **Gemini**, **OpenAI** ou **Anthropic**
    3. Escolha o formato do prompt: **Markdown** ou **⚡ TOON** (econômico)
    4. Faça upload de um ou mais arquivos `.json` ou `.md` de conversas
    5. Clique em **Gerar Resumo** e veja o streaming
    6. Baixe o `.md` ou gere um **prompt de continuidade**
    """)

    st.divider()

    st.markdown("### 🧩 Providers")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🦙 Ollama**
        - Modelos locais
        - Zero custo · Zero API key
        - Privacidade total
        """)
        st.markdown("""
        **🔷 OpenAI**
        - GPT-5.5 · GPT-5.4 · GPT-5.4 Mini
        - Requer API key
        - Fatia premium
        """)
    with col2:
        st.markdown("""
        **🤖 Gemini**
        - 2.0 Flash · 3.x Preview
        - Requer API key
        - Créditos pré-pagos
        """)
        st.markdown("""
        **🧠 Anthropic**
        - Claude Opus 4.8 · Sonnet 4.6 · Haiku 4.5
        - Requer API key
        - Contexto de 1M tokens
        """)

    st.divider()

    st.markdown("### ⚡ TOON — Economia de tokens")

    st.markdown("""
    O projeto suporta **TOON (Token-Oriented Object Notation)**, um formato de prompt
    compacto que elimina emojis, markdown e timestamps — reduzindo **~10-15% de tokens**
    por requisição sem alterar o conteúdo da conversa.

    - **Markdown**: `### 🔥 Desenvolvedor _14:30:00_` → mais legível para a IA
    - **TOON**: `role:user ts:14:30:00` → mais econômico, mesmo resultado
    - Alterne entre os formatos no sidebar do Summarizer
    - Contador de tokens visível antes e durante a geração
    """)

    st.divider()

    st.caption(
        "[GitHub](https://github.com/EzuraWrath/llm-session-summarizer) · "
        "Ollama · Gemini · OpenAI · Anthropic · Streamlit"
    )
