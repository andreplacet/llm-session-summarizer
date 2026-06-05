import streamlit as st

st.set_page_config(
    page_title="LLM Session Summarizer",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed",
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

    st.markdown("""
    Faça upload de arquivos JSON das suas sessões com LLMs (Gemini CLI, OpenCode, etc.)
    e receba um resumo didático em **6 seções** — com streaming em tempo real,
    prompts de continuidade e downloads em markdown.
    """)

    st.divider()

    st.markdown("### 🚀 Como usar")

    st.markdown("""
    1. Vá para **Summarizer** no menu lateral →
    2. Selecione o provedor: **Ollama** (local, gratuito) ou **Gemini** (cloud)
    3. Faça upload de um ou mais arquivos `.json` de conversas
    4. Clique em **Gerar Resumo** e veja o streaming
    5. Baixe o `.md` ou gere um **prompt de continuidade**
    """)

    st.divider()

    st.markdown("### 🧩 Providers")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🦙 Ollama** *(recomendado)*
        - Modelos locais
        - Zero custo
        - Zero API key
        - Privacidade total
        """)
    with col2:
        st.markdown("""
        **🤖 Gemini**
        - Google AI Studio
        - Requer API key
        - Modelos 3.x preview
        - ⚠️ Créditos pré-pagos necessários
        """)

    st.divider()

    st.markdown("### 🔒 Segurança")

    st.markdown("""
    - Chaves de API criptografadas com **PBKDF2 + Fernet AES-128**
    - Senha mestra nunca é armazenada
    - Modo "apenas nesta sessão" disponível
    - LGPD-compliant: eliminação de dados sob demanda
    """)

    st.divider()

    st.caption(
        "[GitHub](https://github.com/andreplacet/llm-session-summarizer) · "
        "Ollama · Gemini AI · Streamlit"
    )
