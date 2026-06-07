import json
import streamlit as st
from pathlib import Path

_TRANSLATIONS: dict[str, dict] = {}

LANG_MAP = {
    "pt": "pt_BR", "pt-BR": "pt_BR", "pt_BR": "pt_BR",
    "en": "en", "en-US": "en", "en-GB": "en",
    "es": "es", "es-ES": "es", "es-MX": "es",
}

FALLBACK = "pt_BR"


def _load(lang: str):
    path = Path(__file__).parent / f"{lang}.json"
    if path.exists():
        _TRANSLATIONS[lang] = json.loads(path.read_text(encoding="utf-8"))


def _resolve(raw: dict, dotted: str):
    keys = dotted.split(".")
    val = raw
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k, {})
        else:
            return dotted
    return val if isinstance(val, str) else dotted


def t(key: str, lang: str | None = None, **kwargs) -> str:
    lang = lang or st.session_state.get("lang") or FALLBACK
    lang = LANG_MAP.get(lang, FALLBACK)
    if lang not in _TRANSLATIONS:
        _load(lang)
    val = _resolve(_TRANSLATIONS.get(lang, {}), key)
    if val == key:
        val = _resolve(_TRANSLATIONS.get(FALLBACK, {}), key)
    try:
        return val.format(**kwargs) if kwargs else val
    except (KeyError, ValueError):
        return val


def detect_language():
    """Inject JS to detect browser language, redirect with ?lang= on first visit."""
    st.html("""
    <script>
    if (!window.location.search.includes('lang=')) {
        var lang = (navigator.language || 'pt-BR').split('-')[0];
        var map = {pt: 'pt_BR', en: 'en', es: 'es'};
        var code = map[lang] || 'pt_BR';
        window.location.search = '?lang=' + code;
    }
    </script>
    """)
