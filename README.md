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

## Setup

```bash
cd ~/dev/personal/python/llm-session-summarizer

# Criar virtualenv
python -m venv .venv && source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# Configurar API key
cp .env.example .env
# Editar .env com sua GEMINI_API_KEY

# Rodar
streamlit run main.py
```

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
    ├── parsers/
    │   ├── base.py            # AbstractParser (interface)
    │   └── gemini_cli.py      # Parser para formato Gemini CLI
    ├── prompts/
    │   └── templates.py       # Templates de prompt (sistema, chunk, merge)
    ├── providers/
    │   ├── base.py            # AbstractProvider (interface)
    │   └── gemini.py          # Provider Gemini (google-genai)
    ├── chunker.py             # Map-reduce para conversas longas
    └── database.py            # SQLite CRUD
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
