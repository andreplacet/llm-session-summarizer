SECTION_HEADERS = {
    "pt_BR": {
        "visao_geral": "1. Visão Geral",
        "topicos": "2. Tópicos Abordados",
        "aprendizados": "3. Aprendizados-Chave",
        "decisoes": "4. Decisões e Encaminhamentos",
        "reflexoes": "5. Reflexões e Pontos Cegos",
        "proximos_passos": "6. Próximos Passos Sugeridos",
    },
    "en": {
        "visao_geral": "1. Overview",
        "topicos": "2. Topics Covered",
        "aprendizados": "3. Key Learnings",
        "decisoes": "4. Decisions & Next Steps",
        "reflexoes": "5. Reflections & Blind Spots",
        "proximos_passos": "6. Suggested Next Steps",
    },
    "es": {
        "visao_geral": "1. Visión General",
        "topicos": "2. Temas Tratados",
        "aprendizados": "3. Aprendizajes Clave",
        "decisoes": "4. Decisiones y Próximos Pasos",
        "reflexoes": "5. Reflexiones y Puntos Ciegos",
        "proximos_passos": "6. Próximos Pasos Sugeridos",
    },
}

OUTPUT_LANG = {
    "pt_BR": "Write the entire summary in Brazilian Portuguese (português do Brasil).",
    "en": "Write the entire summary in English.",
    "es": "Write the entire summary in Spanish (español).",
}


def get_system_prompt(lang: str = "en") -> str:
    output_lang_instruction = OUTPUT_LANG.get(lang, OUTPUT_LANG["en"])
    return f"""You are a didactic mentor specialized in analyzing conversations between developers and AIs.

Your job is to extract knowledge from these conversations, identify patterns, learnings, and produce structured summaries that help the developer consolidate knowledge.

BEFORE producing any answer, internally analyze (without including in the final output) these session metadata:
- Domain/Area: what technical area is being discussed? (e.g., frontend, backend, DevOps, data, mobile, AI/ML, security)
- Role/Profession: what appears to be the developer's profile? (e.g., fullstack dev, data engineer, tech lead, beginner)
- Core theme: what is the main problem or objective?
- Apparent skill level: does the developer demonstrate mastery of the topic or are they learning?
- Specific technologies: which tools, frameworks, languages are involved?

Use this metadata to calibrate each section of the summary:
- DIDACTIC: explain concepts when the apparent skill level indicates a need for fundamentals
- ASSERTIVE: be firm in recommendations when there is technical clarity
- CAUTIOUS: explicitly mark uncertainties and assumptions (e.g., "possibly", "to be confirmed")
- PRUDENT: point out trade-offs, risks, and counterpoints in technical decisions
- CRITICAL: in blind spots and reflections, question assumptions and suggest non-obvious alternatives

Rules:
1. Be concise yet deep — get straight to the point without fluff
2. Identify not only what was said, but what was implicit or assumed
3. Highlight concepts the developer seemed not to fully master
4. Use technical language appropriate to the conversation's level
5. Always format in markdown
6. {output_lang_instruction}"""


def get_chunk_system_prompt(lang: str = "en") -> str:
    output_lang_instruction = OUTPUT_LANG.get(lang, OUTPUT_LANG["en"])
    return f"You are an analyst that summarizes technical conversations. {output_lang_instruction}"


CHUNK_PROMPT = """Analyze the excerpt below from a conversation between a developer and an AI.

Capture the domain/area, apparent skill level, and technologies involved.

Generate a partial summary capturing:
- What topics were discussed in this excerpt
- What the developer asked or requested
- What the AI answered, did, or investigated
- What decisions were made
- What files, code, or configurations were mentioned

Conversation:
{conversation_text}

Partial summary:"""


def get_merge_prompt(lang: str = "en") -> str:
    h = SECTION_HEADERS.get(lang, SECTION_HEADERS["en"])
    return f"""Based on the partial summaries below — each representing a segment of the same session — generate a UNIFIED and STRUCTURED final summary.

The partial summaries are:
{{partial_summaries}}

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


def get_continuity_system_prompt(lang: str = "en") -> str:
    output_lang_instruction = OUTPUT_LANG.get(lang, OUTPUT_LANG["en"])
    return (
        "You are a prompt generator for AI tools. "
        "Your job is to create structured, actionable, and self-contained prompts. "
        f"{output_lang_instruction}"
    )


CONTINUITY_SOURCE_LABEL = {
    "pt_BR": "- **Ferramenta alvo**: {source}",
    "en": "- **Target tool**: {source}",
    "es": "- **Herramienta objetivo**: {source}",
}


def get_continuity_prompt(lang: str, summary: str, source: str = "") -> str:
    source_line = CONTINUITY_SOURCE_LABEL.get(lang, CONTINUITY_SOURCE_LABEL["en"]).format(source=source) if source else ""
    return f"""Based on the summary below of a conversation between a developer and an AI,
generate an OPTIMIZED CONTINUATION PROMPT for use in AI CLIs (Gemini CLI, OpenCode, etc.).

The prompt must follow this structure:

- **Context**: summary of what has already been done (1 paragraph)
- **Objective**: what to do next, clear and specific
- **Constraints**: stack, language, patterns, style (if mentioned)
- **Tasks**: list of concrete, actionable steps
{source_line}

Conversation summary:
{summary}

Generate ONLY the final prompt (no explanations), formatted in markdown.
The prompt must be self-contained — the developer should be able to copy and paste it
into any AI CLI to continue the work immediately."""
