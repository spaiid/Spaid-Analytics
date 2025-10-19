from pydantic import BaseModel

class NormJobPosting(BaseModel):
    dt: str
    platform: str | None
    brand: str | None
    domain: str | None
    posting_id: str | None
    title: str | None
    location: str | None
    department: str | None
    url: str | None
    posted_at: str | None
    updated_at: str | None
    row_id: str
