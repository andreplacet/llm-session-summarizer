import asyncio
import json
import logging
import os
import re
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
from src.prompts.templates import (
    CHUNK_PROMPT,
    OUTPUT_LANG,
    SECTION_HEADERS,
    get_chunk_system_prompt,
    get_continuity_prompt,
    get_continuity_system_prompt,
    get_merge_prompt,
    get_system_prompt,
)
from src.providers.gemini import AVAILABLE_MODELS as GEMINI_MODELS, MODEL_LABELS as GEMINI_LABELS, DEFAULT_MODEL as GEMINI_DEFAULT, GeminiProvider
from src.providers.ollama import OllamaProvider
from src.providers.openai import AVAILABLE_MODELS as OPENAI_MODELS, MODEL_LABELS as OPENAI_LABELS, DEFAULT_MODEL as OPENAI_DEFAULT, OpenAIProvider
from src.providers.anthropic import AVAILABLE_MODELS as ANTHROPIC_MODELS, MODEL_LABELS as ANTHROPIC_LABELS, DEFAULT_MODEL as ANTHROPIC_DEFAULT, AnthropicProvider
from src.database import Database
from src.formatters import FORMATTERS, estimate_tokens
from src.i18n import t

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
        try:
            yield from provider.generate_stream_sync(system_prompt, user_prompt)
        except Exception as exc:
            raise RuntimeError(str(exc)[:200]) from exc
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
            raise RuntimeError(str(value)[:200]) from value
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

lang = st.session_state.get("lang", "en")

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


def _get_direct_prompt(
    lang: str,
    start_time: str,
    last_updated: str,
    msg_count: int,
    conversation_text: str,
) -> str:
    h = SECTION_HEADERS.get(lang, SECTION_HEADERS["en"])
    return f"""Analyze the conversation below between a developer and an AI and generate a structured summary.

**Session metadata:**
- Start: {start_time}
- Last updated: {last_updated}
- Relevant messages: {msg_count}

**Conversation:**

{conversation_text}

Before writing the sections, internally identify the domain, theme, developer role, and technologies involved (do not include this in the output). Use this analysis to calibrate each section with the appropriate tone: didactic where fundamentals are lacking, assertive where there is clarity, cautious with uncertainties, prudent about trade-offs, critical on blind spots.

Generate the final summary EXACTLY with the following sections in markdown:

## {h['visao_geral']}
[2-3 paragraphs summarizing what the entire session was about, the project context, and the developer's main goal]

## {h['topicos']}
[Bulleted list of technical topics discussed, ordered from most to least relevant]

## {h['aprendizados']}
[The most important things learned — concepts, techniques, patterns, best practices, pitfalls avoided]

## {h['decisoes']}
[What was decided, which files were created or modified, which technical direction was taken and why]

## {h['reflexoes']}
[What could have been addressed but wasn't, unconsidered risks, unexplored alternatives, possible improvements in approach]

## {h['proximos_passos']}
[Concrete, actionable recommendations for what to do next, based on what was discussed]

Final summary:"""


def _build_direct_prompt(conversation: ParsedConversation, fmt: str = "markdown", lang: str = "en") -> tuple[str, int]:
    conv_text, conv_tokens = _build_conversation_text(conversation.messages, fmt)
    prompt = _get_direct_prompt(
        lang=lang,
        start_time=conversation.metadata.get("startTime", "N/A"),
        last_updated=conversation.metadata.get("lastUpdated", "N/A"),
        msg_count=len(conversation.messages),
        conversation_text=conv_text,
    )
    total_tokens = estimate_tokens(prompt) + estimate_tokens(get_system_prompt(lang))
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


def _run_chunked_summary(provider, conversation, model_name, fmt: str = "markdown", max_concurrent: int = 5, lang: str = "en") -> str:
    """Process chunks with progress bar, then stream the merge."""
    import asyncio

    from src.chunker import _split_into_chunks

    chunks = _split_into_chunks(conversation)
    total_chunks = len(chunks)

    chunk_system_prompt = get_chunk_system_prompt(lang)

    semaphore = asyncio.Semaphore(max_concurrent)
    completed = {"count": 0}

    async def _summarize_chunk(chunk_conv: ParsedConversation, idx: int) -> str:
        async with semaphore:
            text, _ = _build_conversation_text(chunk_conv.messages, fmt)
            prompt = CHUNK_PROMPT.format(conversation_text=text)
            result = await provider.generate(
                system_prompt=chunk_system_prompt,
                user_prompt=prompt,
            )
            completed["count"] += 1
            s.update(label=t("processing.status_chunk", lang, done=completed["count"], total=total_chunks))
            return f"--- Trecho {idx + 1} de {total_chunks} ---\n{result}"

    async def _run_all() -> list[str]:
        tasks = []
        for i, chunk in enumerate(chunks):
            tasks.append(_summarize_chunk(chunk, i))
        return await asyncio.gather(*tasks)

    with st.status(t("processing.status_start", lang, chunks=total_chunks)) as s:
        partial_summaries = asyncio.run(_run_all())
        s.update(label=t("processing.status_merge", lang), state="running")

    merge_prompt = get_merge_prompt(lang).format(partial_summaries="\n\n".join(partial_summaries))

    response = st.write_stream(
        _async_iter_to_sync(provider, get_system_prompt(lang), merge_prompt)
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
            f"**📂 {t('chat.files_sent', lang, count=len(filenames))}:**\n\n{files_list}\n\n"
            f"_{t('chat.msg_extracted', lang, count=msg_count)}_"
        )

    # Generate summary
    with st.chat_message("assistant"):
        try:
            if msg_count <= CHUNK_SIZE:
                prompt, est_tokens = _build_direct_prompt(conversation, fmt, lang=lang)
                st.caption(t("chat.tokens_estimate", lang, count=est_tokens))
                with st.spinner(t("chat.generating_summary", lang)):
                    response = st.write_stream(
                        _async_iter_to_sync(provider, get_system_prompt(lang), prompt)
                    )
            else:
                response = _run_chunked_summary(
                    provider, conversation, model_name, fmt,
                    max_concurrent=1 if isinstance(provider, OllamaProvider) else 5,
                    lang=lang,
                )
        except Exception as exc:
            import logging
            logging.getLogger("llm_summarizer").error(
                "Erro ao gerar resumo", exc_info=True
            )
            response = t("errors.internal", lang)
            st.error(response)

    # Save to session state
    st.session_state.messages.append(
        {
            "role": "user",
            "content": f"📂 {', '.join(filenames)} ({msg_count} mensagens)",
        }
    )
    st.session_state.messages.append(
        {"role": "assistant", "content": response, "is_prompt": False, "sid": sid, "title": session_title}
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
    # ── Lang selector ──
    lang = st.session_state.get("lang", "en")
    lang_labels = {"pt_BR": "Português", "en": "English", "es": "Español"}
    selected_lang = st.selectbox(
        t("sidebar.lang", lang),
        options=list(lang_labels.keys()),
        format_func=lambda l: lang_labels[l],
        index=list(lang_labels.keys()).index(lang),
    )
    if selected_lang != lang:
        st.session_state["lang"] = selected_lang
        st.query_params["lang"] = selected_lang
        st.rerun()

    st.title(f"🔬 {t('sidebar.title', lang)}")

    dark_mode = st.toggle(f"🌙 {t('sidebar.dark_mode', lang)}", value=True)
    _apply_theme(dark_mode)

    st.divider()

    # ── Provider selector ──
    _ollama_available = _check_ollama_reachable()

    _provider_index = 0 if _ollama_available else 1
    _provider_help = (
        t("sidebar.provider_help_ollama", lang)
    )
    if not _ollama_available:
        _provider_help = t("sidebar.provider_help_cloud", lang)

    provider_name = st.selectbox(
        f"🤖 {t('sidebar.provider', lang)}",
        ["ollama", "gemini", "openai", "anthropic"],
        index=_provider_index,
        help=_provider_help,
    )

    if not _ollama_available:
        st.info(t("sidebar.cloud_info", lang))

    # ── Key management (for providers that need API keys) ──
    if provider_name != "ollama":
        if provider_name == "gemini":
            if not _ollama_available:
                st.warning(t("warnings.gemini_credits_cloud", lang))
            else:
                st.warning(t("warnings.gemini_credits_local", lang))
        st.subheader(f"🔑 {t('key.title', lang)}")

        if _key_is_unlocked(provider_name):
            st.success(f"🔓 {t('key.unlocked', lang)}")
            key_val = st.session_state.get(f"api_key_{provider_name}", "")
            masked = key_val[:4] + "••••" + key_val[-4:] if len(key_val) > 10 else "••••"
            st.caption(f"`{masked}`")
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"🔒 {t('key.lock', lang)}", use_container_width=True, key=f"lock_{provider_name}"):
                    _lock_key(provider_name)
                    st.session_state.pop(f"api_key_{provider_name}", None)
                    st.rerun()
            with col2:
                if st.button(f"🗑️ {t('key.remove', lang)}", use_container_width=True, key=f"delkey_{provider_name}"):
                    _delete_stored_key(provider_name)
                    st.rerun()

        elif _key_is_stored(provider_name):
            st.info(f"🔒 {t('key.stored', lang)}")
            passphrase = st.text_input(
                t("key.passphrase", lang),
                type="password",
                placeholder=t("key.passphrase_placeholder", lang),
                key=f"unlock_{provider_name}",
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"🔓 {t('key.unlock', lang)}", use_container_width=True, key=f"unlockbtn_{provider_name}", disabled=not passphrase):
                    if _unlock_key(provider_name, passphrase):
                        st.session_state.pop(f"unlock_{provider_name}", None)
                        st.rerun()
                    else:
                        st.error(t("key.wrong_password", lang))
            with col2:
                if st.button(f"🗑️ {t('key.remove', lang)}", use_container_width=True, key=f"delkey2_{provider_name}"):
                    _delete_stored_key(provider_name)
                    st.rerun()

        else:
            st.caption(t("key.encrypt_info", lang))
            with st.expander(f"⚙️ {t('key.config_title', lang)}", expanded=not _get_active_api_key(provider_name)):
                api_key_input = st.text_input(
                    t("key.api_key_label", lang),
                    type="password",
                    placeholder=t("key.api_key_placeholder", lang),
                    key=f"apikey_{provider_name}",
                )
                save_mode = st.radio(
                    t("key.save_mode", lang),
                    [f"💾 {t('key.save_encrypted', lang)}", f"⚡ {t('key.save_session', lang)}"],
                    index=0,
                    key=f"savemode_{provider_name}",
                )
                if save_mode.startswith("💾"):
                    master_pass = st.text_input(
                        t("key.passphrase", lang),
                        type="password",
                        placeholder=t("key.create_passphrase", lang),
                        key=f"master_{provider_name}",
                    )
                    if st.button(
                        f"💾 {t('key.save_button', lang)}",
                        use_container_width=True,
                        key=f"savebtn_{provider_name}",
                        disabled=not (api_key_input and master_pass),
                    ):
                        if len(master_pass) < 8:
                            st.error(t("key.passphrase_too_short", lang))
                        else:
                            _save_key(provider_name, api_key_input, master_pass)
                            st.session_state.pop(f"apikey_{provider_name}", None)
                            st.session_state.pop(f"master_{provider_name}", None)
                            st.rerun()
                else:
                    if st.button(
                        f"⚡ {t('key.use_session', lang)}",
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
            model_options = _ollama_models + [f"✏️ {t('sidebar.custom_model', lang)}"]
            default_idx = 0
        else:
            model_options = [t("sidebar.no_models", lang), f"✏️ {t('sidebar.custom_model', lang)}"]
            default_idx = 1
            st.warning(t("sidebar.no_ollama", lang))
    elif provider_name == "openai":
        model_options = [f"{m}  ({OPENAI_LABELS.get(m, '')})" for m in OPENAI_MODELS] + [f"✏️ {t('sidebar.custom_model', lang)}"]
        default_idx = 0
    elif provider_name == "anthropic":
        model_options = [f"{m}  ({ANTHROPIC_LABELS.get(m, '')})" for m in ANTHROPIC_MODELS] + [f"✏️ {t('sidebar.custom_model', lang)}"]
        default_idx = 0
    else:
        model_options = [f"{m}  ({GEMINI_LABELS.get(m, '')})" for m in GEMINI_MODELS] + [f"✏️ {t('sidebar.custom_model', lang)}"]
        default_idx = 0

    model_choice = st.selectbox(f"📊 {t('sidebar.model', lang)}", model_options, index=default_idx)
    if model_choice == f"✏️ {t('sidebar.custom_model', lang)}":
        model_name = st.text_input(
            "Nome do modelo",
            placeholder=t("sidebar.custom_model_placeholder", lang),
            key="custom_model",
            max_chars=100,
        )
    elif model_choice == t("sidebar.no_models", lang):
        model_name = st.text_input(
            "Nome do modelo",
            placeholder="llama3",
            key="custom_model2",
            max_chars=100,
        )
    else:
        model_name = model_choice.split("  ")[0]

    fmt_choice = st.selectbox(
        f"📦 {t('sidebar.format', lang)}",
        options=list(FORMATTERS.keys()),
        format_func=lambda k: FORMATTERS[k].label,
        help=t("sidebar.format_help", lang),
    )

    uploaded_files = st.file_uploader(
        f"📂 {t('sidebar.upload', lang)}",
        type=["json", "md"],
        accept_multiple_files=True,
        help=t("sidebar.upload_help", lang),
    )

    session_title = st.text_input(
        f"📝 {t('sidebar.session_title', lang)}",
        placeholder=t("sidebar.session_placeholder", lang),
        max_chars=200,
    )

    can_process = uploaded_files and (provider_name == "ollama" or _get_active_api_key(provider_name))

    if uploaded_files:
        try:
            for f in uploaded_files:
                f.seek(0)
            conv, _ = _parse_all_files(uploaded_files)
            preview, preview_tokens = _build_conversation_text(conv.messages, fmt_choice)
            st.caption(t("chat.tokens_preview", lang, count=preview_tokens, fmt=FORMATTERS[fmt_choice].label.split()[0]))
        except Exception:
            pass

    process_btn = st.button(
        f"🔍 {t('sidebar.generate', lang)}",
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
        st.warning(t("warnings.select_model", lang))

    st.divider()

    st.subheader(f"📜 {t('sidebar.history_title', lang)}")
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
                            {"role": "assistant", "content": content, "sid": row["id"], "title": row["title"]},
                        ]
                        st.rerun()
            with c2:
                if st.button("🗑️", key=f"del_{row['id']}"):
                    DB.delete_session(row["id"])
                    st.rerun()

    if rows:
        if st.button(f"🗑️ {t('sidebar.history_delete_all', lang)}", use_container_width=True, key="clear_all_history"):
            DB.delete_all_by_session_key(session_key)
            st.rerun()
    else:
        st.caption(t("sidebar.history_empty", lang))

    st.caption(f"🕐 {t('sidebar.history_expiry', lang)}")

# ---------------------------------------------------------------------------
# UI — Main chat area
# ---------------------------------------------------------------------------
st.title(t("sidebar.title", lang))

if "messages" not in st.session_state:
    st.session_state.messages = []


def _safe_filename(title: str, is_prompt: bool = False) -> str:
    sanitized = re.sub(r'[\\/*?:"<>|\n\r\t]', '', title or "").strip()
    sanitized = re.sub(r'\s+', '_', sanitized)[:50].strip('_')
    date_str = datetime.now().strftime("%Y-%m-%d")
    base = sanitized or ("prompt" if is_prompt else "resumo")
    return f"{base}_{date_str}.md"


for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

    if msg["role"] == "assistant" and msg["content"] and not msg["content"].startswith("❌"):
        is_prompt = msg.get("is_prompt", False)
        label = t("chat.download_prompt", lang) if is_prompt else t("chat.download_summary", lang)
        filename = _safe_filename(msg.get("title", ""), is_prompt)

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
                st.caption(f"✅ {t('chat.prompt_generated', lang)}")

    if msg["role"] == "assistant" and not msg.get("is_prompt") and msg["content"] and not msg["content"].startswith("❌"):
        if st.button(f"🤖 {t('chat.generate_prompt', lang)}", key=f"gen_{idx}", use_container_width=True):
            st.session_state["prompt_to_generate"] = idx
            st.rerun()

if st.session_state.messages:
    if st.button(f"🗑️ {t('sidebar.new_summary', lang)}", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if process_btn and uploaded_files:
    import time
    now = time.time()
    last = st.session_state.get("_last_process_time", 0)
    cooldown = 5
    if now - last < cooldown:
        st.warning(t("warnings.rate_limit", lang, seconds=cooldown - int(now - last)))
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
            st.error(t("errors.internal", lang))

if not st.session_state.messages and not uploaded_files:
    st.info(
        t("chat.empty_state", lang)
    )

# Handle prompt generation — runs AFTER messages render, so streaming stays at bottom
if st.session_state.get("prompt_to_generate") is not None:
    _gen_idx = st.session_state["prompt_to_generate"]
    if _gen_idx < len(st.session_state.messages) and provider_name and model_name:
        _summary = st.session_state.messages[_gen_idx]["content"]
        _source = st.session_state.get(f"src_{_gen_idx}", "")
        _provider = _make_provider(provider_name, model_name)

        _prompt_for_llm = get_continuity_prompt(lang, _summary, _source)

        with st.chat_message("assistant"):
            with st.spinner(f"🤖 {t('chat.generating_prompt', lang)}"):
                try:
                    response = st.write_stream(
                        _async_iter_to_sync(
                            _provider,
                            get_continuity_system_prompt(lang),
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
                    st.error(t("errors.internal_prompt", lang))

    st.session_state["prompt_to_generate"] = None
    st.rerun()
