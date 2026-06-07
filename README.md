# LLM Session Summarizer

> Resumos estruturados e didáticos de conversas com IAs —
> suporte a Gemini CLI (`.json`) e OpenCode (`.md`)

## Funcionalidades

### Processamento
- Upload de arquivos **`.json`** (Gemini CLI) e **`.md`** (OpenCode) — múltiplos arquivos simultâneos
- Detecção automática de formato por extensão, dispatch para o parser correto
- Extração inteligente de mensagens relevantes (usuário + IA), ignorando:
  - Gemini CLI: mensagens `info`, `functionResponse`, contextos de sessão
  - OpenCode: seções `_Thinking:_`, blocos de ferramentas (`**Tool:**`, `**Input:**`, `**Output:**`), code fences
- Map-reduce automático para conversas longas (divide em chunks, resume cada um, depois faz merge)
- Processamento paralelo de chunks (Ollama: 1 por vez; Gemini: até 5 concorrentes)

### Resumo e prompts
- Resumo estruturado em **6 seções**: Visão Geral, Tópicos, Aprendizados, Decisões, Reflexões, Próximos Passos
- Streaming da resposta (resumo aparece progressivamente, token por token)
- **🤖 Gerar prompt de continuidade**: botão abaixo de cada resumo que gera um prompt estruturado
  (Contexto → Objetivo → Restrições → Tarefas) otimizado para uso em qualquer CLI de IA
- Campo opcional **📋 Fonte da conversa** (ex: "Gemini CLI", "OpenCode") ao lado de cada resumo

### Download e histórico
- **📥 Baixar resumo `.md`** — download do resumo em markdown, um botão por mensagem
- **📥 Baixar prompt `.md`** — download separado para prompts de continuidade
- Histórico de sessões com SQLite local — recupere resumos anteriores com 1 clique
- Exclusão de sessões do histórico

### Providers
| Provider | Tipo | Custo | API Key | Modelos |
|---|---|---|---|---|
| **🦙 Ollama** | Local | Zero | Não precisa | Qualquer modelo local (`deepseek-v4-flash`, `nemotron-3-super`, etc.) |
| **🤖 Gemini** | Cloud | Créditos pré-pagos | Sim (AI Studio) | `gemini-3.0-flash-preview`, `gemini-3.0-pro-preview` |

### UI/UX
- Tema dark/light com detecção automática do sistema
- CSS reescrito com seletores específicos — contraste legível em ambos os temas
- Sidebar colapsável com seleção de provider, modelo e upload
- Barra de progresso durante processamento de conversas longas
- Indicador de streaming durante geração de resumo e prompt

## 🔐 Segurança e LGPD

### Modelo de criptografia de chaves

Para viabilizar deploy web em conformidade com a LGPD, a chave da API
**nunca é armazenada em texto plano**. O modelo segue o padrão de carteiras
digitais (Metamask, etc.):

```
Usuário digita API Key + cria uma "senha mestra"
    ↓
Senha mestra → PBKDF2 (600k iterações) → chave AES-128
    ↓
API Key é criptografada com Fernet (AES-CBC + HMAC)
    ↓
Apenas o blob criptografado (salt + ciphertext) é salvo no SQLite
    ↓
Na próxima sessão: usuário digita senha mestra → descriptografa
```

**Garantias:**
- O servidor NUNCA vê a chave em texto plano (só recebe o blob criptografado)
- Se o banco de dados vazar, o atacante só obtém blobs ilegíveis sem a senha
- Senha mestra NUNCA é armazenada (nem em hash)
- Modo "apenas nesta sessão": chave existe só em `st.session_state`, perdida ao fechar

**Conformidade LGPD:**
- Dado pessoal (API key) é armazenado com criptografia forte (art. 46)
- Usuário pode remover a chave a qualquer momento (direito à eliminação, art. 18)
- Modo sessão efêmera implementa "right to be forgotten" por design

### Modos de uso

| Modo | Persistência | Segurança |
|---|---|---|
| 💾 Salvar criptografada | SQLite (blob) | Alta — requer senha mestra |
| ⚡ Apenas nesta sessão | `st.session_state` | Máxima — some ao fechar |
| `.env` (fallback) | Arquivo local | Baixa — só para dev local |

## Setup

```bash
cd ~/dev/personal/python/llm-session-summarizer

# Criar virtualenv
python3 -m venv .venv && source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# Para usar Ollama (recomendado):
ollama serve  # em outro terminal

# Para usar Gemini (opcional):
cp .env.example .env
# Editar .env com sua GEMINI_API_KEY

# Rodar
streamlit run main.py
```

> **Recomendação:** Use Ollama com modelos locais (`deepseek-v4-flash`, `nemotron-3-super`, etc.) — zero custo, zero API key, privacidade total.

## Estrutura

```
.
├── main.py                    # Página inicial (landing page)
├── pages/
│   └── 1_Summarizer.py        # App principal (sidebar + chat + processamento)
├── requirements.txt
├── .env                       # API keys (não versionado)
├── .env.example
├── .streamlit/
│   └── config.toml            # Config do Streamlit (tema, layout)
├── data/
│   └── sessions.db            # SQLite (criado automaticamente)
├── docs/
│   └── decisions/             # ADRs (decisões de arquitetura)
└── src/
    ├── models.py              # Dataclasses (Message, ParsedConversation)
    ├── crypto.py              # Criptografia (PBKDF2 + Fernet AES-128)
    ├── database.py            # SQLite CRUD (sessões + chaves criptografadas)
    ├── chunker.py             # Map-reduce: split em chunks, resume, merge
    ├── parsers/
    │   ├── base.py            # AbstractParser — interface Union[dict, str]
    │   ├── gemini_cli.py      # Parser para JSON do Gemini CLI
    │   └── opencode_md.py     # Parser para Markdown do OpenCode
    ├── prompts/
    │   └── templates.py       # Templates: sistema, chunk, merge, prompt de continuidade
    └── providers/
        ├── base.py            # AbstractProvider (interface async)
        ├── gemini.py          # Provider Gemini (google-genai SDK)
        └── ollama.py          # Provider Ollama (HTTP para localhost:11434)
```

## Formatos de entrada suportados

### Gemini CLI (`.json`)

Arquivos exportados pelo Gemini CLI com a estrutura:

```json
{
  "sessionId": "...",
  "kind": "main",
  "startTime": "2026-01-01T00:00:00Z",
  "lastUpdated": "2026-01-01T01:00:00Z",
  "messages": [
    { "type": "user",   "content": [{"text": "..."}], "timestamp": "..." },
    { "type": "gemini",  "content": "...", "thoughts": [...], "toolCalls": [...] },
    { "type": "info",    "content": "..." }
  ]
}
```

O parser (`GeminiCLIParser`) automaticamente:
- Extrai mensagens `user` (entrada do dev) e `gemini` (resposta + raciocínio)
- Inclui nomes das ferramentas usadas como nota inline: `*(Ferramentas: read, edit)*`
- Ignora `info` (sistema), `functionResponse` (resultados de tools), contextos de sessão
- Normaliza para o formato interno `Message(role, text, timestamp)`

### OpenCode (`.md`)

Arquivos Markdown exportados pelo OpenCode com a estrutura:

```markdown
# Título da sessão

**Session ID:** ses_abc123...
**Created:** 6/4/2026, 1:39:48 PM
**Updated:** 6/5/2026, 8:27:46 PM

---

## Assistant (Build · DeepSeek V4 Pro · 4.9s)

Textos e respostas visíveis do assistente...

---

## User

Mensagens do desenvolvedor...

---
```

O parser (`OpenCodeMDParser`) automaticamente:
- Extrai metadata do cabeçalho (título, session ID, timestamps)
- Segmenta mensagens pelos separadores `---`
- Identifica `role` via headings `## Assistant` / `## User`
- **Filtra conteúdo ruidoso:**
  - Seções `_Thinking:_` (raciocínio interno do modelo)
  - Blocos de ferramentas: `**Tool:**`, `**Input:**` (JSON), `**Output:**` (code fences)
  - Code fences remanescentes após limpeza
- Colapsa linhas em branco múltiplas
- Normaliza para o formato interno `Message(role, text)`

## Fluxo de processamento

```
Arquivo(s) enviado(s)
    │
    ├── .json → json.loads() → GeminiCLIParser.can_parse(dict)
    └── .md   → f.read().decode() → OpenCodeMDParser.can_parse(str)
    │
    ▼
ParsedConversation (Message[] + metadata)
    │
    ├── ≤ CHUNK_SIZE mensagens → prompt direto → streaming
    └── > CHUNK_SIZE mensagens → split em chunks
            │
            ├── Chunk 1 → resume (paralelo)
            ├── Chunk 2 → resume (paralelo)
            ├── ...
            └── Merge de todos os resumos parciais → streaming
    │
    ▼
Resposta no chat + botões de download + gerar prompt
```

## Como adicionar um novo provider

1. Criar `src/providers/novo_provider.py` implementando `AbstractProvider`
2. Registrar no seletor de providers em `pages/1_Summarizer.py`
3. Adicionar à função `_make_provider()`

## Como adicionar um novo parser

1. Criar `src/parsers/novo_parser.py` implementando `AbstractParser`:
   - `can_parse(raw: Union[dict, str]) → bool` — detecta se consegue processar
   - `parse(raw: Union[dict, str]) → ParsedConversation` — extrai mensagens e metadata
2. Registrar em `src/parsers/__init__.py`
3. Adicionar à lista `PARSERS` em `pages/1_Summarizer.py`
4. Se for novo formato de arquivo, adicionar extensão ao `file_uploader` (`type=[...]`)

## Dependências

- `streamlit>=1.40.0` — UI web
- `google-genai>=1.0.0` — SDK Gemini
- `python-dotenv>=1.0.0` — .env
- `cryptography>=43.0.0` — criptografia de API keys

Ollama é acessado via HTTP (`localhost:11434`) — **sem dependência Python adicional**.
