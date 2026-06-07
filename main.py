import streamlit as st

st.set_page_config(
    page_title="LLM Session Summarizer",
    page_icon="🔬",
    layout="wide",
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
    Faça upload de arquivos `.json` (Gemini CLI) ou `.md` (OpenCode) das suas sessões com LLMs
    e receba um resumo didático em **6 seções** — com streaming em tempo real,
    prompts de continuidade, downloads em markdown e **formato TOON para economia de tokens**.
    """)

    st.divider()

    st.markdown("### 🚀 Como usar")

    st.markdown("""
    1. Vá para **Summarizer** no menu lateral →
    2. Selecione o provedor: **Ollama** (local, gratuito) ou **Gemini** (cloud)
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
