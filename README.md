# LLM Session Summarizer

> Resumos estruturados e didáticos de conversas com IAs

## Funcionalidades

- Upload de arquivos JSON de sessões com LLMs (Gemini CLI, extensível)
- Extração inteligente de mensagens relevantes (usuário + IA), ignorando ruído de sistema
- Map-reduce automático para conversas longas (evita estouro de tokens)
- Resumo estruturado em 6 seções: Visão Geral, Tópicos, Aprendizados, Decisões, Reflexões, Próximos Passos
- Streaming da resposta (resumo aparece progressivamente)
- Histórico de sessões com SQLite local
- Dark mode
- Providers extensíveis (Gemini hoje, OpenAI/Claude amanhã)

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

### Setup

```bash
cd ~/dev/personal/python/llm-session-summarizer

# Criar virtualenv
python -m venv .venv && source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# Desenvolvimento local (opcional)
cp .env.example .env
# Editar .env com sua GEMINI_API_KEY

# Rodar
streamlit run main.py
```

> Para uso local com `.env`, não configure chave na UI. O app usa `.env` como fallback automaticamente.

## Estrutura

```
.
├── main.py                    # Streamlit app
├── requirements.txt
├── .env                       # API keys (não versionado)
├── .env.example
├── .streamlit/
│   └── config.toml            # Tema escuro, config do Streamlit
├── data/
│   └── sessions.db            # SQLite (criado automaticamente)
├── docs/
│   └── decisions/             # ADRs (decisões de arquitetura)
└── src/
    ├── models.py              # Dataclasses (Message, ParsedConversation)
    ├── crypto.py              # Criptografia (PBKDF2 + Fernet AES-128)
    ├── parsers/
    │   ├── base.py            # AbstractParser (interface)
    │   └── gemini_cli.py      # Parser para formato Gemini CLI
    ├── prompts/
    │   └── templates.py       # Templates de prompt (sistema, chunk, merge)
    ├── providers/
    │   ├── base.py            # AbstractProvider (interface)
    │   └── gemini.py          # Provider Gemini (google-genai)
    ├── chunker.py             # Map-reduce para conversas longas
    └── database.py            # SQLite CRUD (sessões + chaves criptografadas)
```

## Formato de entrada suportado

### Gemini CLI

Arquivos `.json` exportados pelo Gemini CLI com a estrutura:

```json
{
  "sessionId": "...",
  "kind": "main",
  "messages": [
    { "type": "user", "content": [{"text": "..."}] },
    { "type": "gemini", "content": "...", "thoughts": [...] },
    { "type": "info", "content": "..." }
  ]
}
```

O parser automaticamente:
- Extrai mensagens do tipo `user` (entrada do dev) e `gemini` (resposta + raciocínio)
- Ignora `info` (sistema), `functionResponse` (resultados de tools), contextos de sessão
- Normaliza para o formato interno `Message(role, text, timestamp)`

## Como adicionar um novo provider

1. Criar `src/providers/novo_provider.py` implementando `AbstractProvider`
2. Registrar no `main.py`

## Como adicionar um novo parser

1. Criar `src/parsers/novo_parser.py` implementando `AbstractParser`
2. Registrar em `PARSERS` no `main.py`
