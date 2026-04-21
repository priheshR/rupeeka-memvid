import re
from langdetect import detect, LangDetectException

SCRIPT_PATTERNS = {
    'si': re.compile(r'[\u0D80-\u0DFF]'),
    'ta': re.compile(r'[\u0B80-\u0BFF]'),
    'zh': re.compile(r'[\u4E00-\u9FFF]'),
    'ar': re.compile(r'[\u0600-\u06FF]'),
}

def detect_language(text: str) -> str:
    for lang, pattern in SCRIPT_PATTERNS.items():
        if pattern.search(text):
            return lang
    try:
        return detect(text)
    except LangDetectException:
        return 'en'
