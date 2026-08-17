from typing import List

from pydantic import BaseModel, Field, model_validator


class PageRange(BaseModel):
    page: int = Field(ge=1)
    page_end: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_page_range(self):
        if self.page_end is None:
            self.page_end = self.page
        if self.page_end < self.page:
            raise ValueError("page_end cannot be before page")
        return self


class Chunk(PageRange):
    id: int = 0
    document: str
    headings: List[str] = Field(default_factory=list)
    content_types: List[str] = Field(default_factory=list)
    text: str
    embedding: List[float]


class SearchHit(PageRange):
    document: str
    headings: List[str] = Field(default_factory=list)
    content_types: List[str] = Field(default_factory=list)
    text: str
    score: float


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=12)


class Citation(PageRange):
    id: int
    document: str
    section: str = ""
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    grounded: bool
    guardrail_policy: str
    filtered_chunks: int = 0
