"""Base collector interface.

Every job source (Greenhouse, Lever, ...) implements this interface so the
rest of the pipeline can treat all sources the same way: call collect() and
get back a list of normalized Job objects.
"""

from abc import ABC, abstractmethod

from models.job import Job


class BaseCollector(ABC):
    """Contract that every job source collector must follow."""

    @abstractmethod
    def collect(self) -> list[Job]:
        """Fetch raw jobs from the source and return them as normalized Job objects."""
        raise NotImplementedError
