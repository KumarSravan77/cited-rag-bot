import os
from typing import List, Sequence

from app.models import SearchHit
from app.retrieval import HashEmbedder


class OpenAIEmbedder:
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        response = self.client.embeddings.create(model=self.model, input=list(texts))
        return [item.embedding for item in response.data]


def build_embedder():
    return OpenAIEmbedder() if os.getenv("OPENAI_API_KEY") else HashEmbedder()


def synthesize(question: str, hits: List[SearchHit]) -> str:
    if not hits:
        return "I could not find supporting information in the indexed documents."
    context = "\n\n".join(
        f"[{index}] {hit.document}, page {hit.page}: {hit.text}"
        for index, hit in enumerate(hits, start=1)
    )
    if not os.getenv("OPENAI_API_KEY"):
        return f"{hits[0].text} [1]"
    from openai import OpenAI
    response = OpenAI().responses.create(
        model=os.getenv("ANSWER_MODEL", "gpt-5.6-luna"),
        instructions=(
            "Answer only from the supplied sources. Cite every factual claim using [n]. "
            "If the sources are insufficient, say so. Never invent a citation."
        ),
        input=f"Question: {question}\n\nSources:\n{context}",
    )
    return response.output_text
