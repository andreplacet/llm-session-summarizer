import asyncio
import json
import os
import uuid
from datetime import datetime
from queue import Queue
from threading import Thread
from typing import Iterator

import streamlit as st
from dotenv import load_dotenv

from src.chunker import CHUNK_SIZE, summarize_conversation
from src.crypto import encrypt, decrypt
from src.models import ParsedConversation, Message
from src.parsers.gemini_cli import GeminiCLIParser
from src.prompts.templates import SYSTEM_PROMPT
from src.providers.gemini import AVAILABLE_MODELS, GeminiProvider
from src.database import Database

load_dotenv()

# ---------------------------------------------------------------------------
# Parser registry — add new parsers here
# ---------------------------------------------------------------------------
PARSERS: dict[str, GeminiCLIParser] = {
    "gemini_cli": GeminiCLIParser(),
}

# ---------------------------------------------------------------------------
# Page config & theme
# ---------------------------------------------------------------------------
st.set_page_config(page_title="LLM Session Summarizer", page_icon="🔬", layout="wide")


def _apply_theme(dark: bool) -> None:
    bg = "#0e1117" if dark else "#ffffff"
    fg = "#e0e0e0" if dark else "#1a1a1a"
    sbg = "#1a1d23" if dark else "#f0f0f5"
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-color: {bg};
    }}
    .stChatMessage {{
        background-color: {sbg} !important;
        color: {fg} !important;
    }}
    div[data-testid="stFileUploaderDropzone"] {{
        background-color: {sbg};
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Streaming bridge: async generator → sync iterator (thread + queue)
# ---------------------------------------------------------------------------
def _async_iter_to_sync(provider, system_prompt: str, user_prompt: str) -> Iterator[str]:
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
        asyncio.run(_run())

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
KEY_PROVIDER = "gemini"

DB = Database()


def _key_is_unlocked() -> bool:
    return bool(st.session_state.get("api_key"))


def _key_is_stored() -> bool:
    return DB.has_encrypted_key(KEY_PROVIDER)


def _unlock_key(passphrase: str) -> bool:
    encrypted = DB.get_encrypted_key(KEY_PROVIDER)
    if not encrypted:
        return False
    try:
        st.session_state.api_key = decrypt(encrypted, passphrase)
        return True
    except ValueError:
        return False


def _lock_key() -> None:
    st.session_state.api_key = None


def _save_key(api_key: str, passphrase: str) -> None:
    encrypted = encrypt(api_key, passphrase)
    DB.save_encrypted_key(KEY_PROVIDER, encrypted)
    st.session_state.api_key = api_key


def _delete_stored_key() -> None:
    DB.delete_encrypted_key(KEY_PROVIDER)
    st.session_state.api_key = None


def _get_active_api_key() -> str | None:
    return st.session_state.get("api_key") or os.getenv("GEMINI_API_KEY")


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def _build_conversation_text(messages: list[Message]) -> str:
    lines: list[str] = []
    for m in messages:
        role = "🔥 Desenvolvedor" if m.role == "user" else "🤖 IA"
        header = f"### {role}"
        ts = m.timestamp.strftime("%H:%M:%S") if m.timestamp else "??:??:??"
        lines.append(f"{header}  _{ts}_\n{m.text}")
    return "\n\n---\n\n".join(lines)


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


def _build_direct_prompt(conversation: ParsedConversation) -> str:
    return _DIRECT_PROMPT.format(
        start_time=conversation.metadata.get("startTime", "N/A"),
        last_updated=conversation.metadata.get("lastUpdated", "N/A"),
        msg_count=len(conversation.messages),
        conversation_text=_build_conversation_text(conversation.messages),
    )


# ---------------------------------------------------------------------------
# Processing pipeline
# ---------------------------------------------------------------------------
def _parse_all_files(uploaded_files) -> tuple[ParsedConversation, list[str]]:
    all_messages: list[Message] = []
    filenames: list[str] = []
    metadata: dict = {}

    for f in uploaded_files:
        filenames.append(f.name)
        raw = json.loads(f.read())
        f.seek(0)  # reset for potential re-read

        for name, parser in PARSERS.items():
            if parser.can_parse(raw):
                conv = parser.parse(raw)
                all_messages.extend(conv.messages)
                if not metadata:
                    metadata = conv.metadata
                break
        else:
            raise ValueError(
                f"Formato do arquivo '{f.name}' não reconhecido. "
                f"Formatos suportados: {', '.join(PARSERS.keys())}"
            )

    if not all_messages:
        raise ValueError("Nenhuma mensagem relevante encontrada nos arquivos enviados.")

    combined = ParsedConversation(messages=all_messages, metadata=metadata)
    return combined, filenames


def _run_chunked_summary(provider, conversation, model_name) -> str:
    """Process chunks with progress bar, then stream the merge."""
    import asyncio

    from src.chunker import _split_into_chunks
    from src.prompts.templates import CHUNK_PROMPT, MERGE_PROMPT

    chunks = _split_into_chunks(conversation)
    total_chunks = len(chunks)

    progress = st.progress(0, text="Processando trechos da conversa...")
    status = st.empty()

    async def _summarize_chunk(chunk_conv: ParsedConversation, idx: int) -> str:
        text = _build_conversation_text(chunk_conv.messages)
        prompt = CHUNK_PROMPT.format(conversation_text=text)
        result = await provider.generate(
            system_prompt="Você é um analista que resume conversas técnicas. Escreva em português do Brasil.",
            user_prompt=prompt,
        )
        progress.progress(
            (idx + 1) / total_chunks,
            text=f"Processando trecho {idx + 1}/{total_chunks}...",
        )
        return f"--- Trecho {idx + 1} de {total_chunks} ---\n{result}"

    async def _run_all() -> list[str]:
        tasks = []
        for i, chunk in enumerate(chunks):
            tasks.append(_summarize_chunk(chunk, i))
        return await asyncio.gather(*tasks)

    partial_summaries = asyncio.run(_run_all())

    status.info(f"{total_chunks} trechos processados. Gerando resumo final...")

    merge_prompt = MERGE_PROMPT.format(partial_summaries="\n\n".join(partial_summaries))

    response = st.write_stream(
        _async_iter_to_sync(provider, SYSTEM_PROMPT, merge_prompt)
    )

    progress.empty()
    status.empty()

    return response


def _handle_process(
    provider, conversation, filenames, model_name, session_title
) -> None:
    msg_count = len(conversation.messages)

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
                prompt = _build_direct_prompt(conversation)
                response = st.write_stream(
                    _async_iter_to_sync(provider, SYSTEM_PROMPT, prompt)
                )
            else:
                response = _run_chunked_summary(provider, conversation, model_name)
        except Exception as exc:
            response = f"❌ **Erro ao gerar resumo:** {exc}"
            st.error(response)

    # Save to session state
    st.session_state.messages.append(
        {
            "role": "user",
            "content": f"📂 {', '.join(filenames)} ({msg_count} mensagens)",
        }
    )
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Persist to DB
    sid = str(uuid.uuid4())
    DB.save_session(
        id=sid,
        title=session_title or filenames[0],
        source=conversation.metadata.get("source", "unknown"),
        provider=model_name,
        filenames=", ".join(filenames),
        message_count=msg_count,
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

    # ── Key management ──
    st.subheader("🔑 Chave da API")

    if _key_is_unlocked():
        st.success("🔓 Chave desbloqueada")
        masked = st.session_state.api_key[:6] + "•••" + st.session_state.api_key[-4:]
        st.caption(f"`{masked}`")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔒 Bloquear", use_container_width=True):
                _lock_key()
                st.rerun()
        with col2:
            if st.button("🗑️ Remover", use_container_width=True):
                _delete_stored_key()
                st.rerun()

    elif _key_is_stored():
        st.info("🔒 Chave criptografada salva")
        passphrase = st.text_input(
            "Senha mestra",
            type="password",
            placeholder="Digite sua senha para desbloquear",
            key="unlock_passphrase",
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔓 Desbloquear", use_container_width=True, disabled=not passphrase):
                if _unlock_key(passphrase):
                    st.rerun()
                else:
                    st.error("Senha incorreta!")
        with col2:
            if st.button("🗑️ Remover", use_container_width=True):
                _delete_stored_key()
                st.rerun()

    else:
        st.caption(
            "Sua chave é criptografada com uma senha mestra "
            "antes de ser salva. Apenas o blob criptografado "
            "fica armazenado — em conformidade com a LGPD."
        )
        with st.expander("⚙️ Configurar chave", expanded=not _get_active_api_key()):
            api_key_input = st.text_input(
                "API Key",
                type="password",
                placeholder="sk-... ou AIza...",
                key="api_key_input",
            )
            save_mode = st.radio(
                "Modo de armazenamento",
                ["💾 Salvar criptografada (recomendado)", "⚡ Apenas nesta sessão"],
                index=0,
                key="save_mode",
            )
            if save_mode.startswith("💾"):
                master_pass = st.text_input(
                    "Senha mestra",
                    type="password",
                    placeholder="Crie uma senha forte",
                    key="master_pass",
                )
                if st.button(
                    "💾 Salvar chave criptografada",
                    use_container_width=True,
                    disabled=not (api_key_input and master_pass),
                ):
                    if len(master_pass) < 4:
                        st.error("A senha mestra deve ter pelo menos 4 caracteres.")
                    else:
                        _save_key(api_key_input, master_pass)
                        st.rerun()
            else:
                if st.button(
                    "⚡ Usar nesta sessão",
                    use_container_width=True,
                    disabled=not api_key_input,
                ):
                    st.session_state.api_key = api_key_input
                    st.rerun()

    st.divider()

    provider_name = st.selectbox("🤖 Provider", ["gemini"], disabled=True)
    model_name = st.selectbox("📊 Modelo", AVAILABLE_MODELS, index=0)

    uploaded_files = st.file_uploader(
        "📂 Upload JSON(s)",
        type=["json"],
        accept_multiple_files=True,
        help="Formatos suportados: Gemini CLI (.json)",
    )

    session_title = st.text_input(
        "📝 Título da sessão",
        placeholder="Ex: Configuração do Marten Outbox",
    )

    can_process = bool(_get_active_api_key() and uploaded_files)
    process_btn = st.button(
        "🔍 Gerar Resumo",
        type="primary",
        disabled=not can_process,
        use_container_width=True,
    )

    if uploaded_files and not _get_active_api_key():
        st.warning("Configure uma chave de API para continuar.")

    st.divider()

    st.subheader("📜 Histórico")
    for row in DB.get_all_sessions():
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

# ---------------------------------------------------------------------------
# UI — Main chat area
# ---------------------------------------------------------------------------
st.title("Resumo da Conversa")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if process_btn and uploaded_files:
    try:
        provider = GeminiProvider(model=model_name, api_key=_get_active_api_key())
        conversation, filenames = _parse_all_files(uploaded_files)
        _handle_process(provider, conversation, filenames, model_name, session_title)
        st.rerun()
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Erro inesperado: {exc}")

if not st.session_state.messages and not uploaded_files:
    st.info(
        "👈 Faça upload de um ou mais arquivos JSON de conversas com LLMs "
        "e clique em **Gerar Resumo** para começar."
    )
