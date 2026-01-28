import re
import spacy

_nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    doc = _nlp(text)
    tokens = [
        t.lemma_
        for t in doc
        if not t.is_stop and len(t) > 2
    ]

    return " ".join(tokens)
