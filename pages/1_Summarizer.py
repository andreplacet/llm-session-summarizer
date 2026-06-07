import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Iterator

# Ensure repo root is on sys.path (needed for Streamlit Cloud deploy)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from dotenv import load_dotenv

from src.chunker import CHUNK_SIZE, summarize_conversation
from src.crypto import encrypt, decrypt
from src.models import ParsedConversation, Message
from src.parsers.base import AbstractParser
from src.parsers.gemini_cli import GeminiCLIParser
from src.parsers.opencode_md import OpenCodeMDParser
from src.prompts.templates import SYSTEM_PROMPT
from src.providers.gemini import AVAILABLE_MODELS as GEMINI_MODELS, MODEL_LABELS as GEMINI_LABELS, DEFAULT_MODEL as GEMINI_DEFAULT, GeminiProvider
from src.providers.ollama import OllamaProvider
from src.providers.openai import AVAILABLE_MODELS as OPENAI_MODELS, MODEL_LABELS as OPENAI_LABELS, DEFAULT_MODEL as OPENAI_DEFAULT, OpenAIProvider
from src.providers.anthropic import AVAILABLE_MODELS as ANTHROPIC_MODELS, MODEL_LABELS as ANTHROPIC_LABELS, DEFAULT_MODEL as ANTHROPIC_DEFAULT, AnthropicProvider
from src.database import Database
from src.formatters import FORMATTERS, estimate_tokens

load_dotenv()

# ---------------------------------------------------------------------------
# Parser registry — add new parsers here
# ---------------------------------------------------------------------------
PARSERS: list[AbstractParser] = [
    GeminiCLIParser(),
    OpenCodeMDParser(),
]

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------


def _apply_theme(dark: bool) -> None:
    if dark:
        bg = "#0e1117"
        sbg = "#1a1d23"
        fg = "#e0e0e0"
        fg_dim = "#a0a0a0"
        border = "#2e2e3a"
        accent = "#4285f4"
        code_bg = "#1e1e2e"
        code_fg = "#cdd6f4"
        input_bg = "#262730"
        user_bubble = "#1a3a5c"
        assistant_bubble = "#1a1d23"
    else:
        bg = "#fafafa"
        sbg = "#f0f0f5"
        fg = "#1a1a2e"
        fg_dim = "#555555"
        border = "#d0d0d5"
        accent = "#1a73e8"
        code_bg = "#f4f4f8"
        code_fg = "#1a1a2e"
        input_bg = "#ffffff"
        user_bubble = "#e3f0ff"
        assistant_bubble = "#f0f0f5"

    st.markdown(
        f"""
    <style>
    /* ── Base ── */
    .stApp {{
        background-color: {bg};
    }}
    .main .block-container {{
        color: {fg};
        padding: 1rem 2rem 1rem 2rem;
    }}

    /* ── Text defaults ── */
    .stMarkdown, .stMarkdown *, [data-testid="stMarkdownContainer"] * {{
        color: {fg} !important;
    }}
    h1, h2, h3, h4, h5, h6, p, li {{
        color: {fg} !important;
    }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background-color: {sbg};
    }}
    [data-testid="stSidebar"] * {{
        color: {fg} !important;
    }}

    /* ── Selectbox dropdown (portal) ── */
    div[data-baseweb="popover"] li, div[data-baseweb="popover"] div {{
        color: {fg} !important;
        background-color: {sbg} !important;
    }}
    div[data-baseweb="popover"] li:hover {{
        background-color: {border} !important;
    }}

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {{
        background-color: {sbg} !important;
        border: 1px solid {border} !important;
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }}

    /* ── Markdown inside chat ── */
    [data-testid="stChatMessage"] h2 {{
        margin-top: 24px;
        margin-bottom: 12px;
    }}
    [data-testid="stChatMessage"] h3 {{
        margin-top: 20px;
        margin-bottom: 10px;
    }}
    [data-testid="stChatMessage"] p {{
        margin-bottom: 12px;
        line-height: 1.7;
    }}
    [data-testid="stChatMessage"] ul, [data-testid="stChatMessage"] ol {{
        margin-bottom: 16px;
        padding-left: 24px;
    }}
    [data-testid="stChatMessage"] li {{
        margin-bottom: 6px;
    }}

    /* ── Action buttons area ── */
    [data-testid="stChatMessage"] .stButton,
    [data-testid="stChatMessage"] .stDownloadButton {{
        margin-top: 16px;
    }}
    [data-testid="stChatMessage"] .stTextInput {{
        margin-top: 16px;
    }}

    /* ── Code blocks ── */
    code {{
        background-color: {code_bg} !important;
        color: {code_fg} !important;
        padding: 2px 6px;
        border-radius: 4px;
    }}
    pre {{
        background-color: {code_bg} !important;
        border: 1px solid {border} !important;
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
    }}

    /* ── Inputs ── */
    input, textarea, [data-baseweb="select"] > div {{
        background-color: {input_bg} !important;
        color: {fg} !important;
        border-color: {border} !important;
    }}

    /* ── Buttons ── */
    .stButton > button, .stDownloadButton > button {{
        background-color: {sbg} !important;
        color: {fg} !important;
        border: 1px solid {border} !important;
        padding: 8px 16px;
    }}
    .stButton > button[kind="primary"] {{
        background-color: {accent} !important;
        color: #ffffff !important;
    }}
    .stDownloadButton > button:hover {{
        border-color: {accent} !important;
        color: {accent} !important;
    }}

    /* ── File uploader ── */
    [data-testid="stFileUploader"] * {{
        color: {fg} !important;
    }}
    [data-testid="stFileUploaderDropzone"] {{
        background-color: {sbg} !important;
        border-color: {border} !important;
    }}
    [data-testid="stFileUploaderDropzone"] button {{
        color: {fg} !important;
        background-color: {input_bg} !important;
        border-color: {accent} !important;
    }}
    [data-testid="stFileUploaderDropzone"] small {{
        color: {fg_dim} !important;
    }}

    /* ── Alerts / expander ── */
    .stAlert, .streamlit-expanderHeader {{
        background-color: {sbg} !important;
        color: {fg} !important;
    }}
    .streamlit-expanderHeader svg {{
        fill: {fg} !important;
    }}

    /* ── Divider ── */
    hr {{
        border-color: {border} !important;
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Streaming bridge: async generator → sync iterator (thread + queue)
# ---------------------------------------------------------------------------
def _async_iter_to_sync(provider, system_prompt: str, user_prompt: str) -> Iterator[str]:
    if hasattr(provider, "generate_stream_sync"):
        yield from provider.generate_stream_sync(system_prompt, user_prompt)
        return

    q: Queue = Queue()

    async def _run() -> None:
        try:
            async for chunk in provider.generate_stream(system_prompt, user_prompt):
                q.put(("data", chunk))
        except Exception as exc:
            q.put(("error", exc))
        finally:
            q.put(("done", None))

    def _target() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_run())
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    t = Thread(target=_target, daemon=True)
    t.start()

    while True:
        kind, value = q.get()
        if kind == "done":
            break
        elif kind == "error":
            raise value
        else:
            yield value


# ---------------------------------------------------------------------------
# Key management (crypto + session state)
# ---------------------------------------------------------------------------
PROVIDER_ENV_VARS = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

DB = Database()

# Persistent session token (via URL query param, survives tab close)
TOKEN_MAX_AGE = 30  # days

if "session_key" not in st.session_state:
    token = st.query_params.get("s")
    if token and DB.is_valid_token(token, TOKEN_MAX_AGE):
        st.session_state["session_key"] = token
        DB.touch_session_token(token)
    else:
        new_token = str(uuid.uuid4())
        DB.save_session_token(new_token)
        st.session_state["session_key"] = new_token
        st.query_params["s"] = new_token
        st.rerun()


def _key_is_unlocked(provider_name: str) -> bool:
    return bool(st.session_state.get(f"api_key_{provider_name}"))


def _key_is_stored(provider_name: str) -> bool:
    return DB.has_encrypted_key(provider_name)


def _unlock_key(provider_name: str, passphrase: str) -> bool:
    encrypted = DB.get_encrypted_key(provider_name)
    if not encrypted:
        return False
    try:
        st.session_state[f"api_key_{provider_name}"] = decrypt(encrypted, passphrase)
        return True
    except ValueError:
        return False


def _lock_key(provider_name: str) -> None:
    st.session_state[f"api_key_{provider_name}"] = None


def _save_key(provider_name: str, api_key: str, passphrase: str) -> None:
    encrypted = encrypt(api_key, passphrase)
    DB.save_encrypted_key(provider_name, encrypted)
    st.session_state[f"api_key_{provider_name}"] = api_key


def _delete_stored_key(provider_name: str) -> None:
    DB.delete_encrypted_key(provider_name)
    st.session_state.pop(f"api_key_{provider_name}", None)


def _get_active_api_key(provider_name: str) -> str | None:
    key = st.session_state.get(f"api_key_{provider_name}")
    if key:
        return key
    env_var = PROVIDER_ENV_VARS.get(provider_name, "")
    if env_var:
        key = st.secrets.get(env_var) if hasattr(st, "secrets") else None
        if key:
            return key
        key = os.getenv(env_var)
        if key:
            return key
    return None


@st.cache_data(ttl=60)
def _get_ollama_models() -> list[str]:
    return asyncio.run(OllamaProvider.list_models())


@st.cache_data(ttl=300)
def _check_ollama_reachable() -> bool:
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        result = s.connect_ex(("localhost", 11434))
        s.close()
        return result == 0
    except Exception:
        return False


def _make_provider(provider_name: str, model_name: str):
    if provider_name == "ollama":
        return OllamaProvider(model=model_name)
    if provider_name == "openai":
        return OpenAIProvider(model=model_name, api_key=_get_active_api_key("openai"))
    if provider_name == "anthropic":
        return AnthropicProvider(model=model_name, api_key=_get_active_api_key("anthropic"))
    return GeminiProvider(model=model_name, api_key=_get_active_api_key("gemini"))


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def _build_conversation_text(messages: list[Message], fmt: str = "markdown") -> tuple[str, int]:
    formatter = FORMATTERS.get(fmt, FORMATTERS["markdown"])
    text = formatter.format_messages(messages)
    tokens = estimate_tokens(text)
    return text, tokens


_DIRECT_PROMPT = """Analise a conversa abaixo entre um desenvolvedor e uma IA e gere um resumo estruturado.

**Metadados da sessão:**
- Início: {start_time}
- Última atualização: {last_updated}
- Total de mensagens relevantes: {msg_count}

**Conversa:**

{conversation_text}

Gere o resumo final EXATAMENTE com as seguintes seções em markdown:

## 1. Visão Geral
[2-3 parágrafos resumindo sobre o que foi a sessão inteira, o contexto do projeto e o objetivo principal do desenvolvedor]

## 2. Tópicos Abordados
[Lista com bullets dos temas técnicos discutidos, organizados do mais ao menos relevante]

## 3. Aprendizados-Chave
[O que foi aprendido de mais importante — conceitos, técnicas, padrões, boas práticas, armadilhas evitadas]

## 4. Decisões e Encaminhamentos
[O que ficou decidido, quais arquivos foram criados ou modificados, qual direção técnica foi tomada e por quê]

## 5. Reflexões e Pontos Cegos
[O que poderia ter sido abordado e não foi, riscos não considerados, alternativas não exploradas, possíveis melhorias na abordagem]

## 6. Próximos Passos Sugeridos
[Recomendações concretas e acionáveis do que fazer a seguir, baseado no que foi discutido]

Resumo final:"""


def _build_direct_prompt(conversation: ParsedConversation, fmt: str = "markdown") -> tuple[str, int]:
    conv_text, conv_tokens = _build_conversation_text(conversation.messages, fmt)
    formatter = FORMATTERS.get(fmt, FORMATTERS["markdown"])
    meta = formatter.format_metadata(conversation.metadata)
    prompt = _DIRECT_PROMPT.format(
        start_time=conversation.metadata.get("startTime", "N/A"),
        last_updated=conversation.metadata.get("lastUpdated", "N/A"),
        msg_count=len(conversation.messages),
        conversation_text=conv_text,
    )
    total_tokens = estimate_tokens(prompt) + estimate_tokens(SYSTEM_PROMPT)
    return prompt, total_tokens


# ---------------------------------------------------------------------------
# Processing pipeline
# ---------------------------------------------------------------------------
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_TOTAL_MESSAGES = 500


def _parse_all_files(uploaded_files) -> tuple[ParsedConversation, list[str]]:
    all_messages: list[Message] = []
    filenames: list[str] = []
    metadata: dict = {}

    for f in uploaded_files:
        filenames.append(f.name)
        if f.size > MAX_UPLOAD_BYTES:
            raise ValueError(
                f"Arquivo '{f.name}' excede o limite de 50 MB."
            )
        content_bytes = f.read()
        f.seek(0)

        if f.name.endswith(".md"):
            raw = content_bytes.decode("utf-8")
        else:
            raw = json.loads(content_bytes)

        for parser in PARSERS:
            if parser.can_parse(raw):
                conv = parser.parse(raw)
                all_messages.extend(conv.messages)
                if not metadata:
                    metadata = conv.metadata
                break
        else:
            sources = ", ".join(
                type(p).__name__.replace("Parser", "") for p in PARSERS
            )
            raise ValueError(
                f"Formato do arquivo '{f.name}' não reconhecido. "
                f"Formatos suportados: {sources}"
            )

    if not all_messages:
        raise ValueError("Nenhuma mensagem relevante encontrada nos arquivos enviados.")

    if len(all_messages) > MAX_TOTAL_MESSAGES:
        raise ValueError(
            f"Limite de {MAX_TOTAL_MESSAGES} mensagens excedido "
            f"({len(all_messages)} encontradas). Divida em arquivos menores."
        )

    combined = ParsedConversation(messages=all_messages, metadata=metadata)
    return combined, filenames


def _run_chunked_summary(provider, conversation, model_name, fmt: str = "markdown", max_concurrent: int = 5) -> str:
    """Process chunks with progress bar, then stream the merge."""
    import asyncio

    from src.chunker import _split_into_chunks
    from src.prompts.templates import CHUNK_PROMPT, MERGE_PROMPT

    chunks = _split_into_chunks(conversation)
    total_chunks = len(chunks)

    progress = st.progress(0, text="Processando...")
    status = st.empty()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _summarize_chunk(chunk_conv: ParsedConversation, idx: int) -> str:
        async with semaphore:
            text, _ = _build_conversation_text(chunk_conv.messages, fmt)
            prompt = CHUNK_PROMPT.format(conversation_text=text)
            result = await provider.generate(
                system_prompt="Você é um analista que resume conversas técnicas. Escreva em português do Brasil.",
                user_prompt=prompt,
            )
            progress.progress(
                (idx + 1) / total_chunks,
                text=f"Trecho {idx + 1}/{total_chunks}...",
            )
            return f"--- Trecho {idx + 1} de {total_chunks} ---\n{result}"

    async def _run_all() -> list[str]:
        tasks = []
        for i, chunk in enumerate(chunks):
            tasks.append(_summarize_chunk(chunk, i))
        return await asyncio.gather(*tasks)

    partial_summaries = asyncio.run(_run_all())

    status.info(f"{total_chunks} trechos. Gerando resumo...")

    merge_prompt = MERGE_PROMPT.format(partial_summaries="\n\n".join(partial_summaries))

    progress.empty()
    status.empty()

    response = st.write_stream(
        _async_iter_to_sync(provider, SYSTEM_PROMPT, merge_prompt)
    )

    return response


def _handle_process(
    provider, conversation, filenames, model_name, session_title, fmt="markdown"
) -> None:
    msg_count = len(conversation.messages)
    sid = str(uuid.uuid4())

    # Display user "message" bubble
    with st.chat_message("user"):
        files_list = "\n".join(f"- `{fn}`" for fn in filenames)
        st.markdown(
            f"**📂 {len(filenames)} arquivo(s) enviado(s):**\n\n{files_list}\n\n"
            f"_{msg_count} mensagens extraídas_"
        )

    # Generate summary
    with st.chat_message("assistant"):
        try:
            if msg_count <= CHUNK_SIZE:
                prompt, est_tokens = _build_direct_prompt(conversation, fmt)
                st.caption(f"📊 ~{est_tokens} tokens (prompt + sistema)")
                response = st.write_stream(
                    _async_iter_to_sync(provider, SYSTEM_PROMPT, prompt)
                )
            else:
                response = _run_chunked_summary(
                    provider, conversation, model_name, fmt,
                    max_concurrent=1 if isinstance(provider, OllamaProvider) else 5,
                )
        except Exception as exc:
            import logging
            logging.getLogger("llm_summarizer").error(
                "Erro ao gerar resumo", exc_info=True
            )
            response = "Erro interno ao processar. Tente novamente."
            st.error(response)

    # Save to session state
    st.session_state.messages.append(
        {
            "role": "user",
            "content": f"📂 {', '.join(filenames)} ({msg_count} mensagens)",
        }
    )
    st.session_state.messages.append(
        {"role": "assistant", "content": response, "is_prompt": False, "sid": sid}
    )

    # Persist to DB
    DB.save_session(
        id=sid,
        title=session_title or filenames[0],
        source=conversation.metadata.get("source", "unknown"),
        provider=model_name,
        filenames=", ".join(filenames),
        message_count=msg_count,
        session_key=st.session_state.get("session_key", ""),
    )
    DB.save_summary(
        id=str(uuid.uuid4()),
        session_id=sid,
        content=response,
        model_used=model_name,
    )


# ---------------------------------------------------------------------------
# UI — Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔬 Session Summarizer")

    dark_mode = st.toggle("🌙 Dark Mode", value=True)
    _apply_theme(dark_mode)

    st.divider()

    # ── Provider selector ──
    _ollama_available = _check_ollama_reachable()

    _provider_index = 0 if _ollama_available else 1
    _provider_help = (
        "Ollama roda localmente (sem API key). Gemini usa API do Google AI Studio."
    )
    if not _ollama_available:
        _provider_help = (
            "☁️ Ollama indisponível neste ambiente. "
            "Use Gemini (API key) ou execute o projeto localmente para usar Ollama."
        )

    provider_name = st.selectbox(
        "🤖 Provedor",
        ["ollama", "gemini", "openai", "anthropic"],
        index=_provider_index,
        help=_provider_help,
    )

    if not _ollama_available:
        st.info(
            "🖥️ **Ambiente cloud** — Ollama requer servidor local. "
            "Use **Gemini** com API key ou execute o projeto localmente "
            "para usar modelos gratuitos."
        )

    # ── Key management (for providers that need API keys) ──
    if provider_name != "ollama":
        if provider_name == "gemini":
            if not _ollama_available:
                st.warning(
                    "⚠️ **API Gemini requer créditos pré-pagos.** "
                    "Adicione créditos no [AI Studio](https://aistudio.google.com/apikey)."
                )
            else:
                st.warning(
                    "⚠️ **Aviso importante:** O Google descontinuou o tier gratuito da API Gemini. "
                    "Todas as requisições agora exigem **créditos pré-pagos** comprados no "
                    "[AI Studio](https://aistudio.google.com/apikey). "
                    "Recomendamos usar **Ollama** (modelos locais, sem custo)."
                )
        st.subheader("🔑 Chave da API")

        if _key_is_unlocked(provider_name):
            st.success("🔓 Chave desbloqueada")
            key_val = st.session_state.get(f"api_key_{provider_name}", "")
            masked = key_val[:4] + "••••" + key_val[-4:] if len(key_val) > 10 else "••••"
            st.caption(f"`{masked}`")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔒 Bloquear", use_container_width=True, key=f"lock_{provider_name}"):
                    _lock_key(provider_name)
                    st.session_state.pop(f"api_key_{provider_name}", None)
                    st.rerun()
            with col2:
                if st.button("🗑️ Remover", use_container_width=True, key=f"delkey_{provider_name}"):
                    _delete_stored_key(provider_name)
                    st.rerun()

        elif _key_is_stored(provider_name):
            st.info("🔒 Chave criptografada salva")
            passphrase = st.text_input(
                "Senha mestra",
                type="password",
                placeholder="Digite sua senha para desbloquear",
                key=f"unlock_{provider_name}",
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔓 Desbloquear", use_container_width=True, key=f"unlockbtn_{provider_name}", disabled=not passphrase):
                    if _unlock_key(provider_name, passphrase):
                        st.session_state.pop(f"unlock_{provider_name}", None)
                        st.rerun()
                    else:
                        st.error("Senha incorreta!")
            with col2:
                if st.button("🗑️ Remover", use_container_width=True, key=f"delkey2_{provider_name}"):
                    _delete_stored_key(provider_name)
                    st.rerun()

        else:
            st.caption(
                "Sua chave é criptografada com uma senha mestra "
                "antes de ser salva. Apenas o blob criptografado "
                "fica armazenado — em conformidade com a LGPD."
            )
            with st.expander("⚙️ Configurar chave", expanded=not _get_active_api_key(provider_name)):
                api_key_input = st.text_input(
                    "API Key",
                    type="password",
                    placeholder="sk-... ou AIza...",
                    key=f"apikey_{provider_name}",
                )
                save_mode = st.radio(
                    "Modo de armazenamento",
                    ["💾 Salvar criptografada (recomendado)", "⚡ Apenas nesta sessão"],
                    index=0,
                    key=f"savemode_{provider_name}",
                )
                if save_mode.startswith("💾"):
                    master_pass = st.text_input(
                        "Senha mestra",
                        type="password",
                        placeholder="Crie uma senha forte",
                        key=f"master_{provider_name}",
                    )
                    if st.button(
                        "💾 Salvar chave criptografada",
                        use_container_width=True,
                        key=f"savebtn_{provider_name}",
                        disabled=not (api_key_input and master_pass),
                    ):
                        if len(master_pass) < 8:
                            st.error("A senha mestra deve ter pelo menos 8 caracteres.")
                        else:
                            _save_key(provider_name, api_key_input, master_pass)
                            st.session_state.pop(f"apikey_{provider_name}", None)
                            st.session_state.pop(f"master_{provider_name}", None)
                            st.rerun()
                else:
                    if st.button(
                        "⚡ Usar nesta sessão",
                        use_container_width=True,
                        key=f"session_{provider_name}",
                        disabled=not api_key_input,
                    ):
                        st.session_state[f"api_key_{provider_name}"] = api_key_input
                        st.session_state.pop(f"apikey_{provider_name}", None)
                        st.rerun()

    # ── Model selector ──
    st.divider()

    if provider_name == "ollama":
        _ollama_models = _get_ollama_models()
        if _ollama_models:
            model_options = _ollama_models + ["✏️ Outro (digite abaixo)"]
            default_idx = 0
        else:
            model_options = ["Nenhum modelo detectado", "✏️ Outro (digite abaixo)"]
            default_idx = 1
            st.warning(
                "Ollama não detectado. "
                "Se estiver rodando localmente, inicie com `ollama serve`. "
                "Se estiver na versão cloud (demo gratuita), use o provedor Gemini."
            )
    elif provider_name == "openai":
        model_options = [f"{m}  ({OPENAI_LABELS.get(m, '')})" for m in OPENAI_MODELS] + ["✏️ Outro (digite abaixo)"]
        default_idx = 0
    elif provider_name == "anthropic":
        model_options = [f"{m}  ({ANTHROPIC_LABELS.get(m, '')})" for m in ANTHROPIC_MODELS] + ["✏️ Outro (digite abaixo)"]
        default_idx = 0
    else:
        model_options = [f"{m}  ({GEMINI_LABELS.get(m, '')})" for m in GEMINI_MODELS] + ["✏️ Outro (digite abaixo)"]
        default_idx = 0

    model_choice = st.selectbox("📊 Modelo", model_options, index=default_idx)
    if model_choice == "✏️ Outro (digite abaixo)":
        model_name = st.text_input(
            "Nome do modelo",
            placeholder="Ex: llama3 ou gemini-3-flash-preview",
            key="custom_model",
            max_chars=100,
        )
    elif model_choice == "Nenhum modelo detectado":
        model_name = st.text_input(
            "Nome do modelo",
            placeholder="llama3",
            key="custom_model2",
            max_chars=100,
        )
    else:
        model_name = model_choice.split("  ")[0]

    fmt_choice = st.selectbox(
        "📦 Formato do prompt",
        options=list(FORMATTERS.keys()),
        format_func=lambda k: FORMATTERS[k].label,
        help="TOON reduz ~5-15% de tokens removendo formatação visual. "
             "Markdown mantém emojis e timestamps (mais legível para a IA).",
    )

    uploaded_files = st.file_uploader(
        "📂 Upload JSON(s) ou Markdown",
        type=["json", "md"],
        accept_multiple_files=True,
        help="Formatos suportados: Gemini CLI (.json), OpenCode (.md). Limite: 50 MB por arquivo.",
    )

    session_title = st.text_input(
        "📝 Título da sessão",
        placeholder="Ex: Sessão de desenvolvimento",
        max_chars=200,
    )

    can_process = uploaded_files and (provider_name == "ollama" or _get_active_api_key(provider_name))

    if uploaded_files:
        try:
            for f in uploaded_files:
                f.seek(0)
            conv, _ = _parse_all_files(uploaded_files)
            preview, preview_tokens = _build_conversation_text(conv.messages, fmt_choice)
            st.caption(f"📊 ~{preview_tokens} tokens de conversa (formato: {FORMATTERS[fmt_choice].label.split()[0]})")
        except Exception:
            pass

    process_btn = st.button(
        "🔍 Gerar Resumo",
        type="primary",
        disabled=not can_process,
        use_container_width=True,
    )

    if uploaded_files and provider_name != "ollama" and not _get_active_api_key(provider_name):
        provider_labels = {"gemini": "Gemini", "openai": "OpenAI", "anthropic": "Anthropic"}
        provider_label = provider_labels.get(provider_name, "API")
        env_key = PROVIDER_ENV_VARS.get(provider_name, "API_KEY")
        st.warning(
            f"Configure uma chave de API para usar o provedor {provider_label}. "
            + (
                f"No Streamlit Cloud, adicione `{env_key}` em App Settings → Secrets."
                if not _ollama_available
                else "Use o campo 🔑 Chave da API acima ou configure no .env."
            )
        )
    if uploaded_files and provider_name == "ollama" and not model_name:
        st.warning("Selecione ou digite um modelo Ollama.")

    st.divider()

    st.subheader("📜 Histórico")
    # Purge expired sessions before displaying
    session_key = st.session_state.get("session_key", "")
    DB.purge_expired_sessions(session_key, TOKEN_MAX_AGE)
    rows = DB.get_all_sessions(session_key=session_key)
    if rows:
        for row in rows:
            c1, c2 = st.columns([4, 1])
            with c1:
                label = row["title"]
                if len(label) > 40:
                    label = label[:40] + "..."
                if st.button(
                    label, key=f"hist_{row['id']}", use_container_width=True
                ):
                    content = DB.get_summary(row["id"])
                    if content:
                        st.session_state.messages = [
                            {
                                "role": "user",
                                "content": (
                                    f"📂 {row['filenames']} ({row['message_count']} mensagens)"
                                ),
                            },
                            {"role": "assistant", "content": content},
                        ]
                        st.rerun()
            with c2:
                if st.button("🗑️", key=f"del_{row['id']}"):
                    DB.delete_session(row["id"])
                    st.rerun()

    if rows:
        if st.button("🗑️ Apagar todo o histórico", use_container_width=True, key="clear_all_history"):
            DB.delete_all_by_session_key(session_key)
            st.rerun()
    else:
        st.caption("Nenhum resumo gerado nesta sessão.")

    st.caption(
        "🕐 O histórico expira automaticamente após 30 dias. "
        "Faça download dos resumos que quiser guardar antes desse prazo."
    )

# ---------------------------------------------------------------------------
# UI — Main chat area
# ---------------------------------------------------------------------------
st.title("Resumo da Conversa")

if "messages" not in st.session_state:
    st.session_state.messages = []

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

    if msg["role"] == "assistant" and msg["content"] and not msg["content"].startswith("❌"):
        is_prompt = msg.get("is_prompt", False)
        label = "📥 Baixar prompt .md" if is_prompt else "📥 Baixar resumo .md"

        sanitized = msg["content"].replace("`", "").replace("#", "").replace("*", "").strip()
        filename = (sanitized[:40] + "...") if len(sanitized) > 40 else sanitized
        filename = (filename or ("prompt" if is_prompt else "resumo")) + ".md"

        col1, col2 = st.columns([1, 2])
        with col1:
            st.download_button(
                label,
                data=msg["content"],
                file_name=filename,
                mime="text/markdown",
                key=f"dl_{idx}",
                use_container_width=True,
            )
        with col2:
            if not is_prompt:
                st.text_input(
                    "📋 Fonte da conversa (opcional)",
                    key=f"src_{idx}",
                    placeholder="Ex: Gemini CLI, OpenCode, VS Code chat...",
                    label_visibility="collapsed",
                )
            if is_prompt:
                st.caption("✅ Prompt de continuidade gerado")

    if msg["role"] == "assistant" and not msg.get("is_prompt") and msg["content"] and not msg["content"].startswith("❌"):
        if st.button("🤖 Gerar prompt de continuidade", key=f"gen_{idx}", use_container_width=True):
            st.session_state["prompt_to_generate"] = idx
            st.rerun()

if st.session_state.messages:
    if st.button("🗑️ Novo resumo", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if process_btn and uploaded_files:
    import time
    now = time.time()
    last = st.session_state.get("_last_process_time", 0)
    cooldown = 5
    if now - last < cooldown:
        st.warning(f"Aguarde {cooldown - int(now - last)}s antes de gerar outro resumo.")
    else:
        st.session_state["_last_process_time"] = now
        try:
            if provider_name == "ollama":
                provider = OllamaProvider(model=model_name)
            elif provider_name == "openai":
                provider = OpenAIProvider(model=model_name, api_key=_get_active_api_key("openai"))
            elif provider_name == "anthropic":
                provider = AnthropicProvider(model=model_name, api_key=_get_active_api_key("anthropic"))
            else:
                provider = GeminiProvider(model=model_name, api_key=_get_active_api_key("gemini"))
            conversation, filenames = _parse_all_files(uploaded_files)
            _handle_process(provider, conversation, filenames, model_name, session_title, fmt_choice)
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            import logging
            logging.getLogger("llm_summarizer").error(
                "Erro inesperado ao processar", exc_info=True
            )
            st.error("Erro interno ao processar. Tente novamente.")

if not st.session_state.messages and not uploaded_files:
    st.info(
        "👈 Faça upload de um ou mais arquivos (.json ou .md) de conversas com LLMs "
        "e clique em **Gerar Resumo** para começar."
    )

# Handle prompt generation — runs AFTER messages render, so streaming stays at bottom
if st.session_state.get("prompt_to_generate") is not None:
    _gen_idx = st.session_state["prompt_to_generate"]
    if _gen_idx < len(st.session_state.messages) and provider_name and model_name:
        _summary = st.session_state.messages[_gen_idx]["content"]
        _source = st.session_state.get(f"src_{_gen_idx}", "")
        _provider = _make_provider(provider_name, model_name)

        _source_line = f"- **Ferramenta alvo**: {_source}" if _source else ""

        _prompt_for_llm = f"""Com base no resumo abaixo de uma conversa entre desenvolvedor e IA,
gere um PROMPT DE CONTINUAÇÃO otimizado para uso em CLIs de IA (Gemini CLI, OpenCode, etc.).

O prompt deve seguir esta estrutura:

- **Contexto**: resumo do que já foi feito (1 parágrafo)
- **Objetivo**: o que fazer a seguir, claro e específico
- **Restrições**: stack, linguagem, padrões, estilo (se mencionado)
- **Tarefas**: lista de passos concretos e acionáveis
{_source_line}

Resumo da conversa:
{_summary}

Gere APENAS o prompt final (sem explicações), formatado em markdown.
O prompt deve ser autocontido — o dev deve conseguir copiá-lo e colar
em qualquer CLI de IA para continuar o trabalho imediatamente."""

        with st.chat_message("assistant"):
            with st.spinner("🤖 Gerando prompt de continuidade..."):
                try:
                    response = st.write_stream(
                        _async_iter_to_sync(
                            _provider,
                            "Você é um gerador de prompts para IAs. "
                            "Seu trabalho é criar prompts estruturados, acionáveis e autocontidos. "
                            "Escreva em português do Brasil.",
                            _prompt_for_llm,
                        )
                    )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response, "is_prompt": True}
                    )
                    # Save prompt to DB (linked to same session as the summary)
                    _parent_msg = st.session_state.messages[_gen_idx]
                    _parent_sid = _parent_msg.get("sid")
                    if _parent_sid:
                        DB.save_summary(
                            id=str(uuid.uuid4()),
                            session_id=_parent_sid,
                            content=response,
                            model_used=model_name,
                            type="prompt",
                        )
                except Exception as exc:
                    import logging
                    logging.getLogger("llm_summarizer").error(
                        "Erro ao gerar prompt de continuidade", exc_info=True
                    )
                    st.error("Erro interno ao gerar prompt. Tente novamente.")

    st.session_state["prompt_to_generate"] = None
    st.rerun()
