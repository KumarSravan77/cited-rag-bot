from __future__ import annotations

from typing import Iterable, List, Tuple

from guardrails import scan

from app.models import SearchHit


UNSAFE_RETRIEVAL_CATEGORIES = {"prompt_injection", "harmful_instruction", "secret"}


def filter_retrieved_context(hits: Iterable[SearchHit]) -> Tuple[List[SearchHit], int]:
    """Remove retrieved chunks that could instruct the model or leak secrets."""
    safe: List[SearchHit] = []
    filtered = 0
    for hit in hits:
        result = scan(hit.text)
        categories = {finding.category for finding in result.findings}
        if categories & UNSAFE_RETRIEVAL_CATEGORIES:
            filtered += 1
            continue
        # PII is sanitized before context reaches the model while provenance remains intact.
        safe.append(hit.model_copy(update={"text": result.text}))
    return safe, filtered
