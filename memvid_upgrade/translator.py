import re
import os
from typing import List, Dict, Optional
from google import genai
from memvid_upgrade.lang_detect import detect_language

LANG_NAMES = {
    "si": "Sinhala",
    "ta": "Tamil",
    "en": "English",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "ar": "Arabic",
    "es": "Spanish",
    "ja": "Japanese",
}

class GeminiTranslator:
    """Gemini-powered translator with entity protection.
    
    Designed for high-quality Sinhala and Tamil translation
    where open-source models fall short.
    """

    def __init__(
        self,
        target_langs: List[str] = None,
        model: str = "gemini-2.5-pro",
        api_key: str = None,
    ):
        self.target_langs = target_langs or ["si", "ta"]
        self.model = model
        self.client = genai.Client(
            api_key=api_key or os.environ["GOOGLE_API_KEY"]
        )

        # Protect capitalized named entities
        self.entity_re = re.compile(
            r'\b([A-Z][a-zA-Z0-9]+(?:\s[A-Z][a-zA-Z0-9]+){0,3})\b'
        )

    def _extract_entities(self, text: str) -> list:
        """Ask Gemini to identify genuine named entities only.
        
        Returns a list of strings that should NOT be translated:
        - People names (Alice, Bob)
        - Organization names (Anthropic, Google)
        - Product/project names (Alpha, GPT-4)
        - Place names (Colombo, London)
        - Technical identifiers (API names, model names)
        
        Excludes: normal capitalized words at sentence start,
        common nouns, adjectives, verbs.
        """
        prompt = f"""List only the genuine named entities in this text that should NOT be translated.
Include: people names, organization names, product names, project names, place names, brand names, technical identifiers.
Exclude: regular words that happen to be capitalized (sentence starts, common nouns).

Text: {text}

Reply with ONLY a JSON array of strings. Example: ["Alice", "Anthropic", "Alpha"]
If no named entities, reply: []"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        
        try:
            import json, re
            raw = response.text.strip()
            # Extract JSON array from response
            match = re.search(r'\[.*?\]', raw, re.DOTALL)
            if match:
                entities = json.loads(match.group())
                # Only keep entities that actually appear in the text
                return [e for e in entities if e in text]
        except Exception:
            pass
        return []

    def _protect_entities(self, text: str) -> tuple:
        """Replace named entities with placeholders before translation."""
        entities = self._extract_entities(text)
        mapping = {}
        guarded = text
        # Sort by length descending to avoid partial replacements
        for entity in sorted(entities, key=len, reverse=True):
            ph = f"ENTITY{len(mapping):04d}"
            mapping[ph] = entity
            guarded = guarded.replace(entity, ph)
        return guarded, mapping

    def _restore_entities(self, text: str, mapping: dict) -> str:
        for ph, orig in mapping.items():
            text = text.replace(ph, orig)
        return text

    def translate_one(self, text: str, target_lang: str) -> str:
        """Translate text to target language using Gemini."""
        lang_name = LANG_NAMES.get(target_lang, target_lang)
        guarded, mapping = self._protect_entities(text)

        prompt = f"""Translate the following text to {lang_name}.

Rules:
- Output ONLY the translated text, nothing else
- Do not translate words in ALL_CAPS that look like placeholders (e.g. ENTITY0000) — keep them exactly as-is
- Preserve all numbers, currency symbols, and punctuation
- Keep proper nouns and technical terms accurate

Text to translate:
{guarded}"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        translated = response.text.strip()
        return self._restore_entities(translated, mapping)

    def translate_all(
        self,
        text: str,
        source_lang: Optional[str] = None
    ) -> Dict[str, str]:
        """Returns {lang_code: translated_text} for all target langs."""
        src = source_lang or detect_language(text)
        results = {src: text}
        for lang in self.target_langs:
            if lang != src:
                print(f"  Translating to {LANG_NAMES.get(lang, lang)}...")
                results[lang] = self.translate_one(text, lang)
        return results


# Singleton accessor
_translator: Optional[GeminiTranslator] = None

def get_translator(
    target_langs: List[str] = None,
) -> GeminiTranslator:
    global _translator
    if _translator is None:
        _translator = GeminiTranslator(
            target_langs=target_langs or ["si", "ta"]
        )
    return _translator
