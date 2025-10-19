from pydantic import BaseModel, Field

class RawJobPosting(BaseModel):
    source: str
    platform: str | None = None
    fetched_at: str         # ISO UTC
    dt: str                 # partition date (trading session)
    brand: str | None = None
    domain: str | None = None
    posting_id: str | None = None
    title: str | None = None
    location: str | None = None
    url: str | None = None
    department: str | None = None
    posted_at: str | None = None   # original posting timestamp if available
    updated_at: str | None = None  # last updated timestamp if available
    raw: dict = Field(default_factory=dict)
