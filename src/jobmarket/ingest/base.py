"""Abstract base for job board ingest sources."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from pydantic import BaseModel, ConfigDict


class RawRecord(BaseModel):
    """A single raw job payload from an external source."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    payload: dict[str, object]


class AbstractSource(ABC):
    """Fetches raw job records from an external API without parsing."""

    name: str

    @abstractmethod
    def fetch(self) -> AsyncIterator[RawRecord]:
        """Yield raw records as they are fetched from the source."""
        ...
