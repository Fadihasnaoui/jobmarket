"""Abstract base for source-specific job parsers."""

from abc import ABC, abstractmethod
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ParsedJob(BaseModel):
    """Normalized job fields extracted from a raw source payload."""

    model_config = ConfigDict(frozen=True)

    company_name: str
    title: str
    description: str
    location_raw: str | None = None
    country: str | None = None
    city: str | None = None
    is_remote: bool | None = None
    contract_type: str | None = None
    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str | None = None
    salary_period: str | None = None
    salary_is_predicted: bool = False
    posted_at: datetime | None = None
    url: str | None = None


class AbstractParser(ABC):
    """Parses a raw source payload into normalized job fields."""

    source: str

    @abstractmethod
    def parse(self, payload: dict[str, object]) -> ParsedJob:
        """Parse one raw payload into a ParsedJob."""
