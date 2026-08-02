from typing import List

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: int = 0
    document: str
    page: int = Field(ge=1)
    text: str
    embedding: List[float]


class SearchHit(BaseModel):
    document: str
    page: int
    text: str
    score: float


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=12)


class Citation(BaseModel):
    id: int
    document: str
    page: int
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    grounded: bool
