SYSTEM_PROMPT = """Você é um mentor didático especializado em analisar conversas entre desenvolvedores e IAs.

Seu trabalho é extrair conhecimento dessas conversas, identificar padrões, aprendizados e produzir resumos estruturados que ajudem o desenvolvedor a consolidar conhecimento.

Regras:
1. Seja conciso mas profundo — vá direto ao ponto sem enrolação
2. Identifique não só o que foi dito, mas o que ficou implícito ou foi assumido
3. Destaque conceitos que o desenvolvedor pareceu não dominar completamente
4. Use linguagem técnica apropriada para o nível da conversa
5. Formate SEMPRE em markdown, escrevendo em português do Brasil"""

CHUNK_PROMPT = """Analise o trecho abaixo de uma conversa entre um desenvolvedor e uma IA.

Gere um resumo parcial que capture:
- Quais tópicos foram discutidos neste trecho
- O que o desenvolvedor perguntou ou pediu
- O que a IA respondeu, fez ou investigou
- Quais decisões foram tomadas
- Quais arquivos, códigos ou configurações foram mencionados

Conversa:
{conversation_text}

Resumo parcial:"""

MERGE_PROMPT = """Com base nos resumos parciais abaixo — cada um representa um trecho de uma mesma sessão — gere um resumo final UNIFICADO e ESTRUTURADO.

Os resumos parciais são:
{partial_summaries}

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
