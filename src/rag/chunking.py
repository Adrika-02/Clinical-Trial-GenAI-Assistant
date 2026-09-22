"""Token-aware, sentence-boundary-respecting text chunker. Sentences are the
atomic unit — a chunk never splits mid-sentence — and are packed greedily up
to chunk_size tokens, with the last `overlap` tokens' worth of sentences
repeated at the start of the next chunk for retrieval continuity.
"""
import re

import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def split_sentences(text: str) -> list:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list:
    """Pack sentences into chunks of at most chunk_size tokens.

    Args:
        text: source document text.
        chunk_size: max tokens per chunk.
        overlap: approx tokens of trailing context repeated into the next chunk.

    Returns:
        list of chunk strings, in order. A single sentence longer than
        chunk_size becomes its own (oversized) chunk rather than being cut
        mid-sentence.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    sentences = split_sentences(text)
    if not sentences:
        return []

    sentence_tokens = [count_tokens(s) for s in sentences]

    chunks = []
    current = []
    current_tokens = 0
    i = 0
    while i < len(sentences):
        sent, n_tok = sentences[i], sentence_tokens[i]
        if current and current_tokens + n_tok > chunk_size:
            chunks.append(" ".join(current))
            # carry trailing sentences worth ~overlap tokens into next chunk
            carry, carry_tokens = [], 0
            for s, t in zip(reversed(current), reversed([count_tokens(c) for c in current])):
                if carry_tokens + t > overlap:
                    break
                carry.insert(0, s)
                carry_tokens += t
            current = carry
            current_tokens = carry_tokens
            continue
        current.append(sent)
        current_tokens += n_tok
        i += 1

    if current:
        chunks.append(" ".join(current))

    return chunks
