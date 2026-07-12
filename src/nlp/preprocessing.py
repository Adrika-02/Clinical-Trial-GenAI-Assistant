"""SpaCy text preprocessing utilities: tokenization, lemmatization, and
stopword removal for clinical note text."""
import spacy

_NLP = None


def get_nlp():
    """Lazily load the spaCy pipeline (loaded once per process)."""
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def preprocess_text(text: str, nlp=None) -> str:
    """Tokenize, lowercase, lemmatize, and strip stopwords/punctuation.

    Returns a cleaned string of space-joined lemmas suitable for TF-IDF
    vectorization (keeps negation words like 'no'/'not' since they are
    clinically meaningful for adverse-event detection).
    """
    nlp = nlp or get_nlp()
    doc = nlp(text)
    keep_stopwords = {"no", "not", "none", "never"}
    tokens = [
        tok.lemma_.lower()
        for tok in doc
        if not tok.is_punct
        and not tok.is_space
        and (not tok.is_stop or tok.lemma_.lower() in keep_stopwords)
        and tok.is_alpha
    ]
    return " ".join(tokens)


def preprocess_batch(texts, nlp=None) -> list:
    nlp = nlp or get_nlp()
    keep_stopwords = {"no", "not", "none", "never"}
    cleaned = []
    for doc in nlp.pipe(texts, batch_size=64):
        tokens = [
            tok.lemma_.lower()
            for tok in doc
            if not tok.is_punct
            and not tok.is_space
            and (not tok.is_stop or tok.lemma_.lower() in keep_stopwords)
            and tok.is_alpha
        ]
        cleaned.append(" ".join(tokens))
    return cleaned


if __name__ == "__main__":
    sample = "Patient reports mild nausea following Day 14 dose. BP slightly elevated at 138/88."
    print(preprocess_text(sample))
