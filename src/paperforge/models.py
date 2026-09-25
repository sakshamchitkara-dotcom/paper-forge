"""Core data types shared across the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Paper:
    arxiv_id: str  # versionless, e.g. "1412.6980"
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: str  # ISO-8601
    updated: str = ""
    version: str = "v1"
    comment: str = ""
    journal_ref: str = ""
    doi: str = ""
    # enrichment (optional)
    code_url: str = ""
    hf_upvotes: int | None = None
    citations: int | None = None
    sources: list[str] = field(default_factory=list)

    @property
    def abs_url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"

    @property
    def pdf_url(self) -> str:
        return f"https://arxiv.org/pdf/{self.arxiv_id}{self.version}"

    @property
    def html_url(self) -> str:
        return f"https://arxiv.org/html/{self.arxiv_id}{self.version}"

    @property
    def slug(self) -> str:
        return self.arxiv_id.replace("/", "_")

    def citation(self) -> str:
        """Plain-text citation. Always shown next to any output about this paper."""
        authors = ", ".join(self.authors[:6]) + (" et al." if len(self.authors) > 6 else "")
        year = self.published[:4]
        return f"{authors} ({year}). {self.title}. arXiv:{self.arxiv_id}. {self.abs_url}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Paper":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})
