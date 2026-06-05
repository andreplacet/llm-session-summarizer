# ADR 001 — Arquitetura do LLM Session Summarizer

**Status:** aceito
**Data:** 2026-06-05

## Contexto

Precisamos de uma ferramenta para fazer upload de arquivos JSON de conversas com LLMs
e gerar resumos didáticos estruturados usando a API do Gemini. Os arquivos podem conter
centenas de mensagens e ultrapassar os limites de tokens das APIs.

## Decisões

### 1. Streamlit como framework de UI

**Decisão:** Usar Streamlit ao invés de FastAPI + React ou Flask + Jinja.

**Justificativa:**
- Prototipagem rápida — sem HTML/CSS/JS separado
- Componentes nativos de chat (`st.chat_message`, `st.chat_input`)
- Suporte a streaming via `st.write_stream`
- Ampla adoção para dashboards e ferramentas de IA
- Python puro, sem necessidade de build frontend

**Trade-offs:**
- Menos controle sobre o layout que React
- Não é adequado para apps multi-página complexos
- State management manual via `st.session_state`

### 2. Parser extensível com auto-detecção

**Decisão:** Interface `AbstractParser` com método `can_parse()` para detecção
automática do formato.

**Justificativa:**
- Cada LLM exporta conversas em formato diferente
- Auto-detecção evita que o usuário tenha que escolher o formato manualmente
- Fácil adicionar suporte a ChatGPT, Claude e outros no futuro

**Implementação:**
- `GeminiCLIParser` — detecta `sessionId` + `kind` + `messages[]` no root
- Registro em dicionário no `main.py` — sem reflection ou auto-discovery
- Cada parser retorna `list[Message]` padronizado independente da origem

### 3. Map-reduce para conversas longas

**Decisão:** Dividir conversas longas (> 10 mensagens) em chunks com overlap,
resumir cada chunk, depois fazer merge dos resumos parciais.

**Justificativa:**
- APIs LLM têm limite de tokens de input (Gemini Flash: ~1M, mas latência sobe)
- Conversas típicas têm 50-200+ mensagens com contexto extenso
- Map-reduce é o padrão da indústria (LangChain, LlamaIndex usam abordagem similar)
- Permite processamento paralelo dos chunks (`asyncio.gather`)

**Parâmetros:**
- `CHUNK_SIZE = 10` mensagens por chunk
- `CHUNK_OVERLAP = 2` mensagens entre chunks consecutivos (evita perda de contexto)

**Trade-offs:**
- Custo maior (N+1 chamadas à API ao invés de 1)
- Pequena perda de contexto entre chunks não adjacentes
- Overlap mitiga perda de continuidade

### 4. Streaming via thread + queue

**Decisão:** Usar thread + `queue.Queue` para converter o async generator do SDK
Google em um sync iterator compatível com `st.write_stream`.

**Justificativa:**
- O SDK `google-genai` é async-only para streaming (`generate_content_stream`)
- `st.write_stream` espera um iterador síncrono
- Thread separada executa o event loop do asyncio; a main thread consome da queue
- Sem dependências extras (não precisa de `nest-asyncio` ou hacks de event loop)

**Trade-offs:**
- Overhead de thread por request (aceitável para app single-user)
- Exceções precisam ser propagadas manualmente pela queue

### 5. SQLite para persistência

**Decisão:** SQLite local (arquivo `data/sessions.db`) para histórico de sessões.

**Justificativa:**
- Zero dependências externas (não precisa de Postgres, Redis, etc.)
- Suficiente para uso single-user local
- WAL mode para leitura concorrente
- Schema simples: 2 tabelas (`sessions`, `summaries`)

**Trade-offs:**
- Não escala para múltiplos usuários simultâneos
- Sem backup automático

### 6. Seções fixas de resumo no prompt

**Decisão:** O prompt força 6 seções fixas em markdown no resumo final.

**Justificativa:**
- Consistência entre sessões — facilita comparação
- Cobre o ciclo completo de análise: contexto → tópicos → aprendizado → decisões → reflexão → ação
- Markdown é renderizado nativamente pelo Streamlit

**Seções obrigatórias:**
1. Visão Geral
2. Tópicos Abordados
3. Aprendizados-Chave
4. Decisões e Encaminhamentos
5. Reflexões e Pontos Cegos
6. Próximos Passos Sugeridos

### 7. Temperatura baixa para resumos

**Decisão:** `temperature = 0.3` para todas as chamadas de resumo.

**Justificativa:**
- Resumos precisam ser factuais e consistentes
- Baixa criatividade reduz alucinações
- Consistência entre execuções — mesmo input gera output similar

### 8. Providers extensíveis

**Decisão:** Interface `AbstractProvider` com métodos `generate()` e `generate_stream()`.

**Justificativa:**
- Trocar de provider requer apenas implementar a interface
- Dropdown no Streamlit permite troca futura sem alterar lógica de negócio
- Cada provider encapsula seu próprio SDK e autenticação

## Consequências

**Positivas:**
- Arquitetura modular facilita manutenção e extensão
- Streaming melhora UX perceptivelmente (resposta aparece conforme gerada)
- Map-reduce viabiliza conversas arbitrariamente longas

**Negativas:**
- Map-reduce custa mais tokens (N chamadas ao invés de 1)
- Thread + queue adiciona complexidade ao código de streaming
- Sem testes automatizados na versão inicial

## Alternativas consideradas

| Alternativa | Motivo da rejeição |
|---|---|
| FastAPI + React | Complexidade desnecessária para app single-page |
| LangChain | Overkill — só precisamos de map-reduce e prompt templates |
| Resumir apenas últimas N mensagens | Perde contexto valioso do início da conversa |
| Enviar tudo de uma vez (sem chunking) | Estoura tokens em conversas longas |
| JSON estruturado como saída | Menos legível e natural que markdown |
